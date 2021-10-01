import logging
import azure.functions as func
import json
import ast
# Import helper script
# from .predict import predict_image_from_url
from .server_main import runner

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def main(req: func.HttpRequest) -> func.HttpResponse:
    video_url = req.params.get('video')
    queue_polygon = req.params.get('queue_polygon')
    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        video_url = req_body.get('video')
        queue_polygon = req_body.get('queue_polygon')
        
    if not (video_url and queue_polygon):
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a video url and a queue polygon in the query string or in the request body for a personalized response.",
             status_code=200
        )

    queue_polygon = ast.literal_eval(queue_polygon)
    # model = req.params.get('model')
    logging.info('Video URL received: ' + str(video_url))    
    #results = predict_image_from_url(video_url)
    results = runner(video_url, queue_polygon)
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    return func.HttpResponse(json.dumps(results), headers = headers)

