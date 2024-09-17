#!/bin/sh

# Trust policy JSON content
TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

ROLE_NAME=SageMakerExecutionRoleTest

# Check if the role already exists
SAGEMAKER_ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null)

if [ -z "$SAGEMAKER_ROLE_ARN" ]; then
    # Role does not exist; create it
    SAGEMAKER_ROLE_ARN=$(aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document "$TRUST_POLICY" \
        --query 'Role.Arn' --output text)
else
    # Role exists; update the trust policy (if needed)
    aws iam update-assume-role-policy \
        --role-name $ROLE_NAME \
        --policy-document "$TRUST_POLICY" >/dev/null
fi

# Attach policies (suppress output)
aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess >/dev/null

aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess >/dev/null

aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly >/dev/null

# Output only the Role ARN
echo "$SAGEMAKER_ROLE_ARN"