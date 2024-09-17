#!/bin/bash

# Set environment variables
export MODEL_ID="microsoft/Phi-3.5-vision-instruct"
export INSTANCE_TYPE="ml.g6.4xlarge"
export API_HOST="0.0.0.0"
export API_PORT="8001"
export UVICORN_LOG_LEVEL="info"

# Check if required Python packages are installed
REQUIRED_PKG=("vllm" "fastapi" "uvicorn[standard]" "pydantic")
echo "Checking required Python packages..."
for PKG in "${REQUIRED_PKG[@]}"; do
    if ! python -c "import ${PKG%%[*]}" &> /dev/null; then
        echo "Package $PKG is not installed. Installing..."
        pip install "$PKG"
    else
        echo "Package $PKG is already installed."
    fi
done

# Start the sagemaker_serving.py script in the background
echo "Starting the sagemaker_serving.py server..."
python ../src/sagemaker_serving.py &
SERVER_PID=$!

# Function to clean up background process on exit
cleanup() {
    echo "Stopping the server..."
    kill $SERVER_PID
    rm -f test_input.json
    exit
}

# Trap EXIT signal to ensure cleanup
trap cleanup EXIT

# Wait for the server to start
echo "Waiting for the server to start..."
sleep 60

# Check if the server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "Server is running (PID: $SERVER_PID)."
else
    echo "Server failed to start."
    exit 1
fi

# Create test input JSON file
cat > test_input.json << EOF
{
  "model": "microsoft/Phi-3.5-vision-instruct",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ]
}
EOF

# Send test request to /ping endpoint
echo "Testing /ping endpoint..."
PING_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$API_PORT/ping)
if [ "$PING_RESPONSE" -eq 200 ]; then
    echo "Ping successful."
else
    echo "Ping failed with status code $PING_RESPONSE."
    cleanup
fi

# Send test request to /invocations endpoint
echo "Sending test request to /invocations endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:$API_PORT/invocations \
     -H "Content-Type: application/json" \
     -d @test_input.json)

echo "Response from server:"
echo "$RESPONSE"

# Optionally, you can parse and display specific fields from the response
# For example, display the assistant's reply:
ASSISTANT_REPLY=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
if [ -n "$ASSISTANT_REPLY" ]; then
    echo "Assistant's reply: $ASSISTANT_REPLY"
else
    echo "Failed to parse assistant's reply."
fi

# Clean up
echo "Cleaning up..."
cleanup

echo "Test completed successfully."