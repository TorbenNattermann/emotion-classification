import boto3
import json

ADA002_ENCODER_LAMBDA = 'dm-ares-ada-002-inference-lambda-stage'


def infere_ada_002(input_text):
    body = {"text_batch": input_text}
    payload = json.dumps(body).encode()
    result = boto3.client('lambda').invoke(
        FunctionName=ADA002_ENCODER_LAMBDA,
        InvocationType='RequestResponse',
        LogType='None',
        Payload=payload
    )
    output = json.loads(result['Payload'].read())
    return output['embeddings']


if __name__ == "__main__":
    texts = ["Andreas Scheuer: Maut operator contradicts the Minister of Transport", "French Open: Alexander Zverev trembles into the third round"]
    print(infere_ada_002(texts))
