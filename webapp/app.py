from flask import Flask, escape, request, render_template
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import wikipedia

app = Flask(__name__)

@app.route('/')
def plot_pred():
	result = {}
	return render_template('plot_pred.html', result= result)

@app.route('/movie_predictor')
def movie_predictor():
	result = {}
	return render_template('movie_name_pred.html', result= result)

@app.route('/predict', methods=['GET', 'POST'])
def predict_plot():
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
	data['blob'] = blob[:100]
	return data

@app.route('/predict_movie_name', methods=['GET', 'POST'])
def predict_movie_name():
	movie_name = request.form.get('movie_name')
	try:
		movie = wikipedia.page(movie_name)
		blob = movie.section('Plot')
		lr_classifier = pickle.load(open('models/lr_classifier.pkl', 'rb'))
		tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
		multilabel_binarizer = pickle.load(open('models/multilabel_binarizer.pkl', 'rb'))
		text = " ".join(blob)
		t = tfidf_vectorizer.transform([blob])
		y_pred_lr = lr_classifier.predict(t)
		print(multilabel_binarizer.inverse_transform(y_pred_lr)[0])
		data = {}
		data['tags'] = list(multilabel_binarizer.inverse_transform(y_pred_lr)[0])
		data['blob'] = blob[:100]
		data['movie_name'] = movie_name
		return data
	except:
		data = {}
		data['tags'] = ["movie Not found"]
		data['blob'] = "movie Not found"
		data['movie_name'] = movie_name
		return data

if __name__ == '__main__':
	app.run(debug=True)