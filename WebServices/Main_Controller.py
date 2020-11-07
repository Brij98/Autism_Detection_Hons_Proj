from flask import Flask

from WebServices.TrainClassifiers_Controller import train_models_bp
from WebServices.ModelPerformance_Controller import model_performance_bp
from WebServices.Classification_Controller import classify_samples_bp

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

app.register_blueprint(train_models_bp, url_prefix='/train_classifiers')
app.register_blueprint(model_performance_bp, url_prefix='/model_performance')
app.register_blueprint(classify_samples_bp, url_prefix='/classify_samples')

if __name__ == "__main__":
    app.run(threaded=True, debug=True)
