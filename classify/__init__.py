import logging
import azure.functions as func
import json
# Import helper script
# from .predict import predict_image_from_url
from .server_main import runner

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def main(req: func.HttpRequest) -> func.HttpResponse:
    video_url = req.params.get('video')
    model = req.params.get('model')
    logging.info('Video URL received: ' + video_url)    
    #results = predict_image_from_url(video_url)
    results = runner(video_url)
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    return func.HttpResponse(json.dumps(results), headers = headers)

