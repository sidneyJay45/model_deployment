from flask import Flask, render_template , request , jsonify
import pickle 

tokenizer = pickle.load(open('models/cv.pkl', 'rb'))
model = pickle.load(open( 'models/clf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods= ['POST','GET'])
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get email content from the form
    email_text = request.form.get('content')

    # Tokenize the email content
    tokenized_email = tokenizer.transform([email_text])

    # Make predictions using the model
    predictions = model.predict(tokenized_email)

    # Convert predictions to 1 or -1 based on the result
    predictions = 1 if predictions == 1 else -1

    # Render the home.html template with predictions and email_text
    return render_template('home.html', predictions=predictions, email_text=email_text)



# we've created an api route
@app.route('/api/predict', methods=['POST'])
def api_predict():
   data = request.get_json(force = True)
   email_text = data['content']
   email_text = request.get_json(force=True)
   # Tokenize the email content
   tokenized_email = tokenizer.transform([email_text])

   # Make predictions using the model
   predictions = model.predict(tokenized_email)

   # Convert predictions to 1 or -1 based on the result
   predictions = 1 if predictions == 1 else -1
   return jsonify ({predictions: predictions})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
