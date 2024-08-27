import boto3

s3 = boto3.client('s3')
BUCKET_NAME = 'ba-torben-nattermann'


def list_buckets(n=10):
    response = s3.list_buckets()
    print('Existing buckets:')
    for bucket in response['Buckets']:
        print(f'  {bucket["Name"]}')


def upload_to_bucket(file_path, object_key):
    """
    Helper class to upload data to bucket
    :param file_path: local file path to be uploaded
    :param object_key: Storage position (folder/obj_name)
    """
    s3.upload_file(file_path, BUCKET_NAME, object_key)
    print(f"File uploaded to S3 bucket: {BUCKET_NAME}, with key: {object_key}")


if __name__ == "__main__":
    upload_to_bucket(file_path='../data/NRC/nrc_dict.pkl', object_key='NRC/nrc_dict.pkl')
    upload_to_bucket(file_path='../data/NRC/lex2.pkl', object_key='NRC/lex_extended.pkl')
