from flask import Flask

from WebServices.TrainClassifiers_Controller import train_models_bp

app = Flask(__name__)

app.register_blueprint(train_models_bp, url_prefix='/train_classifiers')

if __name__ == "__main__":
    app.run(threaded=True)
