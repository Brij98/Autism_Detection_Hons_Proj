from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World"


@app.route('/upload_data_file', methodes=['POST'])
def upload_data_file():
    pass


if __name__ == "__main__":
    app.run(debug=True)
