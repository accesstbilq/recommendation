from flask import Flask, render_template, request
import pickle as pickle
import numpy as np

app = Flask(__name__)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open('new_df.pkl', 'rb') as f:
    new_df = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    input_movie = ''
    
    movie_titles = new_df['title'].tolist()
    
    if request.method == 'POST':
        input_movie = request.form.get('movie', '').strip()
        
        try:
            index = new_df[new_df['title'].str.lower() == input_movie.lower()].index[0]
            
            distances = sorted(list(enumerate(cosine_sim[index])),
                               reverse=True,
                               key=lambda x: x[1])
            
            recommendations = [
                new_df.iloc[i[0]].title
                for i in distances[1:6]
            ]
        
        except (IndexError, KeyError):
            recommendations = ["Movie not found in database"]
        
    return render_template('index.html', 
                           recommendations=recommendations,
                           input_movie=input_movie,
                           movie_titles=movie_titles)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7676)

