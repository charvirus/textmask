from flask import Flask

app = Flask(__name__)

app.debug = True

@app.route("/")
def main():
    return "Matt Damon"

@app.route("/check/<sentence>")
def check_sentence(sentence):

    return sentence

if __name__ == "__main__":
    app.run()