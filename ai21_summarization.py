import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

text_to_summarize = """
France is a country located in Western Europe. It is known for its rich history, culture, and cuisine. The capital city of France is Paris, which is famous for landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. France is also renowned for its contributions to art, fashion, and philosophy. The country has a diverse landscape that includes beautiful coastlines, picturesque countryside, and majestic mountains like the Alps and Pyrenees. French is the official language, and the country has a population of approximately 67 million people. France is a member of the European Union and plays a significant role in global politics and economics.
"""

prompt = f"Summarize the following text in one sentence:\n\n{text_to_summarize}\n\nSummary:"

body = json.dumps({
    "messages" : [{"role": "user", "content": prompt}],
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
})

response = bedrock.invoke_model(
      modelId="ai21.jamba-1-5-large-v1:0",
      body=body
)

response_body = json.loads(response['body'].read())
print(response_body['choices'][0]['message']['content'])