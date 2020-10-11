import boto3
from botocore.client import Config

ACCESS_KEY_ID = ''
ACCESS_SECRET_KEY = ''
BUCKET_NAME = ''


data = open('7b82749466e76b11ca11e6e85eec6225.3.jpg', 'rb')

s3 = boto3.resource(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET_KEY,
    config=Config(signature_version='s3v4')
)
s3.Bucket(BUCKET_NAME).put_object(Key='7b82749466e76b11ca11e6e85eec6225.3.jpg', Body=data)


def moderate_image(photo, bucket):

    client=boto3.client('rekognition')

    response = client.detect_moderation_labels(Image={'S3Object':{'Bucket':bucket,'Name':photo}})
    #print(response)
    print('Detected labels for ' + photo)    
    for label in response['ModerationLabels']:
        print (label['Name'] + ' : ' + str(label['Confidence']))
        #print (label['ParentName'])
    return len(response['ModerationLabels'])



def main():
    photo='7b82749466e76b11ca11e6e85eec6225.3.jpg'
    bucket=''
    label_count=moderate_image(photo, bucket)
    print("Labels detected: " + str(label_count))




main()



