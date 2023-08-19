"""
@title

@description

"""
import flask
from markupsafe import escape

app = flask.Flask(__name__)


@app.route('/')
def index():
    return 'Index Page'


@app.route('/hello')
def hello_default():
    return 'Hello, World'


@app.route('/me', methods=['POST', 'GET'])
def me_api():
    # user = get_current_user()
    return {
        "username": 'Andrew',
        "theme": 'black',
        "image": 'black_image',
    }


# @app.route("/users")
# def users_api():
#     users = get_all_users()
#     return [user.to_json() for user in users]
@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!"


if __name__ == '__main__':
    app.run(debug=True)
