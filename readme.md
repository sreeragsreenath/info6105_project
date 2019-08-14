# Introduction & Background

Classifying a movie plot into genres was chosen as it provides a wide range of exploratory paths with data science methods and its application can be found in various sophisticated recommendation engines. The project aims at exploring various classifier algorithms, understanding their behaviors and enhancing the classifier accuracy to predict the genre. 

  

Our goal with the project is to:

  

1. Conduct EDA on the Data 
2. Learn and Apply NLP on the plot content 
3. Test out various classic and deep machine learning models 
4. Build a pipeline for the best result and pickle the models 
5. Develop a simple to use web application as an API for new classification 
  
  

## Data Source and Description

The data is taken from Kaggle data source :

[https://www.kaggle.com/jrobischon/wikipedia-movie-plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)

#### Content

The dataset contains descriptions of 34,886 movies from around the world. Column descriptions are listed below:

- Release Year - Year in which the movie was released 
- Title - Movie title 
- Origin/Ethnicity - Origin of movie (i.e. American, Bollywood, Tamil, etc.) 
- Director - Director(s) 
- Genre - Movie Genre(s) 
- Wiki Page - URL of the Wikipedia page from which the plot description was scraped 
- Plot - Long form description of movie plot 

#### Acknowledgements

This data was scraped from Wikipedia


### Exploratory Data Analysis

Exploratory Data Analysis (EDA) is an approach for data analysis that employs a variety of techniques (mostly graphical) to :

1. maximize insight into a data set 
2. uncover underlying structure 
3. extract important variables 
4. detect outliers and anomalies 
5. test underlying assumptions 
6. develop parsimonious models  
7. determine optimal factor setting 


  

### Natural Language Processing (NLP)

  

Natural language processing (NLP) is a subfield of [computer science](https://en.wikipedia.org/wiki/Computer_science), [information engineering](https://en.wikipedia.org/wiki/Information_engineering_(field)), and [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence) concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of [natural language](https://en.wikipedia.org/wiki/Natural_language) data.

  

Library Used : Natural Language Toolkit

  
  

### Steps Followed

1. Convert movie plot into lower case 
2. Remove stop words 
3. Stemming  
4. Lemmatization