import asyncio
import os
import sys

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.logger import init_logger
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

# Initialize logger
logger = init_logger('sagemaker_serving')

# Mapping of AWS instance types to the number of GPUs
instance_to_gpus = {
    "ml.g5.4xlarge": 1,
    "ml.g6.4xlarge": 1,
    "ml.g5.12xlarge": 4,
    "ml.g6.12xlarge": 4,
    "ml.g5.48xlarge": 8,
    "ml.g6.48xlarge": 8,
    "ml.p4d.24xlarge": 8,
    "ml.p4de.24xlarge": 8,
    "ml.p5.48xlarge": 8,
}

def get_num_gpus(instance_type):
    try:
        return instance_to_gpus[instance_type]
    except KeyError:
        raise ValueError(f"Instance type {instance_type} not found in the dictionary")

# Global variables for the engine and the OpenAI serving chat instance
async_engine_client: AsyncLLMEngine
openai_serving_chat: OpenAIServingChat

async def create_app():
    # Use make_arg_parser to create the parser with all expected arguments
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    
    # Prepare a list of arguments to pass to parser.parse_args()
    # We'll use environment variables or default values
    cli_args = []
    
    # Set model
    model_id = os.getenv('MODEL_ID', None)
    if model_id is None:
        sys.exit("MODEL_ID must be provided")
    cli_args.extend(['--model', model_id])
    
    # Set tokenizer if needed (optional)
    tokenizer = os.getenv('TOKENIZER', None)
    if tokenizer:
        cli_args.extend(['--tokenizer', tokenizer])
    
    # Set tensor_parallel_size based on instance type
    instance_type = os.getenv('INSTANCE_TYPE', 'ml.g6.4xlarge')
    tensor_parallel_size = get_num_gpus(instance_type)
    cli_args.extend(['--tensor-parallel-size', str(tensor_parallel_size)])
    
    # Set other arguments from environment variables or defaults
    # For example, set host and port
    host = os.getenv('API_HOST', '0.0.0.0')
    port = os.getenv('API_PORT', '8000')
    cli_args.extend(['--host', host, '--port', port])
    
    # Set logging level
    uvicorn_log_level = os.getenv('UVICORN_LOG_LEVEL', 'info')
    cli_args.extend(['--uvicorn-log-level', uvicorn_log_level])
    
    # Add --trust-remote-code flag
    cli_args.append('--trust-remote-code')

    # Add --max-model-len flag
    cli_args.extend(['--max-model-len', '4049'])

    # Add --limit-mm-per-prompt image=2
    cli_args.extend(['--limit-mm-per-prompt', 'image=2'])
    
    # Parse the arguments
    args = parser.parse_args(cli_args)
    
    logger.info(f"Starting SageMaker vLLM with args: {args}")
    
    # Create engine arguments
    engine_args = AsyncEngineArgs.from_cli_args(args)
    
    # Initialize the asynchronous engine client
    global async_engine_client
    async_engine_client = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Get the model configuration
    model_config = await async_engine_client.get_model_config()
    
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    
    request_logger = None if args.disable_log_requests else RequestLogger(max_log_len=args.max_log_len)
    
    # Initialize the OpenAI serving chat instance
    global openai_serving_chat
    openai_serving_chat = OpenAIServingChat(
        async_engine_client,
        model_config,
        served_model_names,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
    )
    
    # Create the FastAPI app
    app = FastAPI()
    
    # Define the /ping endpoint required by SageMaker
    @app.get("/ping")
    async def ping():
        return JSONResponse(content={}, status_code=status.HTTP_200_OK)
    
    # Define the /invocations endpoint required by SageMaker
    @app.post("/invocations")
    async def invocations(request: Request):
        try:
            payload = await request.json()
            chat_completion_request = ChatCompletionRequest(**payload)
        except Exception as e:
            return JSONResponse(content={"error": "Invalid request format", "details": str(e)}, status_code=400)
    
        generator = await openai_serving_chat.create_chat_completion(
            chat_completion_request, request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
    
        if 'stream' in payload and payload['stream']:
            return StreamingResponse(content=generator,
                                     media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())
    
    return app

def start_api_server():
    # Since we need to run async code, we use asyncio.run
    app = asyncio.run(create_app())
    
    # Retrieve host and port from arguments or environment variables
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    uvicorn_log_level = os.getenv('UVICORN_LOG_LEVEL', 'info')
    
    # Spin up the API server
    uvicorn.run(app, host=host, port=port, log_level=uvicorn_log_level)

if __name__ == "__main__":
    start_api_server()