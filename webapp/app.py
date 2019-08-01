from flask import Flask, escape, request, render_template
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

@app.route('/')
def hello():
	result = {}
	return render_template('index.html', result= result)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	blob = request.form.get('movie_plot')
	lr_classifier = pickle.load(open('models/lr_classifier.pkl', 'rb'))
	tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
	multilabel_binarizer = pickle.load(open('models/multilabel_binarizer.pkl', 'rb'))
	text = " ".join(blob)
	t = tfidf_vectorizer.transform([blob])
	y_pred_lr = lr_classifier.predict(t)
	print(multilabel_binarizer.inverse_transform(y_pred_lr)[0])
	data = {}
	data['tags'] = list(multilabel_binarizer.inverse_transform(y_pred_lr)[0])
	return render_template('index.html',result = data)

if __name__ == '__main__':
	app.run(debug=True)