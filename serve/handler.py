import os
import torch
import logging
import json
from model import LinearRegressionModel


class ModelHandler(object):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

        # log_format = '%(name)s - %(levelname)s - %(message)s'
        log_location = os.getenv('LOG_LOCATION')
        print("STEVE DEBUG LOG LOCATION: ", log_location)
        # log_file = os.path.join(log_location, 'handler.log')
        # logging.basicConfig(filename=log_file, filemode='w', format=log_format, level=logging.INFO)
        self.logger = logging.getLogger('handler')

    def initialize(self, context):
        """
        Invoked by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.logger.info("model_dir: %s", model_dir)
        print("STEVE DEBUG MODEL DIR: ", model_dir)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        print("STEVE DEBUG MODEL PT PATH: ", model_pt_path)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = LinearRegressionModel()
        self.model.load_state_dict(torch.load(str(model_pt_path)))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        preprocessed_data = data[0].get("body")
        print("STEVE DEBUG PREPROCESS: ", preprocessed_data)
        return float(preprocessed_data.get("data"))

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output.tolist()
        return postprocess_output

    def handle(self, data, context):
        """
        Invoked by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediction output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """

        print("STEVE DEBUG DATA: ", data)

        model_input = self.preprocess(data)
        tensor_input = torch.tensor([[model_input]])
        model_output = self.model.forward(tensor_input)
        return self.postprocess(model_output)
