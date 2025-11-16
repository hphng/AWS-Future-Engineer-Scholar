import boto3
import json

messages = []

def add_message(role: str, prompt_text: str) -> None:
    """
    Add a message to the conversation history
    role: The role of the message sender (e.g., "user", "assistant")
    prompt_text: The text content of the message
    """
    messages.append({"role": role, "content": [{"type": "text", "text": prompt_text}]})

def invoke_claude_model(prompt_text: str) -> str:
    """
    Send a prompt to claude ad return generated text
    prompt_text: The text prompt to send to the model
    """
    #add user prompt everytime called
    add_message("user", prompt_text)

    #call bedrock model
    session = boto3.Session(profile_name="udacity")
    client = session.client("bedrock-runtime", region_name="us-east-1")
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "messages": messages
    }

    try: 
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        response_body = json.loads(response['body'].read())
        generated_text = response_body['content'][0]['text']

        #add assistant response to conversation history
        add_message("assistant", generated_text)

        return generated_text
    except Exception as e:
        print("Error invoking model:", e)
        return None


def main():
    """
    Main function to run the conversation loop
    """
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        response = invoke_claude_model(user_input)
        if response:
            print("Model response:")
            print(response)

if __name__ == "__main__":
    main()