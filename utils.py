import pickle


tokenizer = pickle.load(open('models/cv.pkl', 'rb'))
model = pickle.load(open( 'models/clf.pkl', 'rb'))



tokenized_email = tokenizer.transform([email_text])

# Make predictions using the model
predictions = model.predict(tokenized_email)

# Convert predictions to 1 or -1 based on the result
predictions = 1 if predictions == 1 else -1

# Render the home.html template 