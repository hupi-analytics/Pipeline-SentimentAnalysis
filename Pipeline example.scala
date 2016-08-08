// Sentiment Analysis

// Package 
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
import java.text.Normalizer

// Data loading
val path = "hdfs://hupi.node1.pro.hupi.loc/user/anthony.laffond/SentimentAnalysis/"

val texte_clean = sc.textFile(path + "CommentairesLivres5000.csv").
  map(l => ( l.substring(0, l.length-10) ,
  if(l.contains("|Positive|")) 1.0 else if (l.contains("|Negative|")) 0.0 else 3.0) )
  //map(l=>l._2).groupBy(l=>l).map(l=> (l._1,l._2.size)).take(10)

// We choose around 2000 reviews with approximatly 50% positive and 50% negative.
val texte_todelete = texte_clean.filter(l=> l._2==1.0).
                                 randomSplit(Array(0.7,0.3),seed=123)(0)
val texte = texte_clean.subtract(texte_todelete).persist()
val n = texte.count()

// Conversion in dataframe
val texte_df = texte.toDF("text","label")
texte_df.show(2)

// Split data into training (80%) and test (20%)
val splits_df = texte_df.randomSplit(Array(0.8, 0.2), seed = 1234)
val training_df = splits_df(0).cache()
val test_df = splits_df(1)
val n_training_df = training_df.count()
val n_test_df = test_df.count()
// Positive/Negative
//training_df.map(l=>l(1)).groupBy(l=>l).map(l=> (l._1,l._2.size)).take(5)

// Split the text into Array
val tokenizer = new Tokenizer().
  setInputCol("text").
  setOutputCol("words")

// TF-IDF
val hashingTF = new HashingTF().
  //setNumFeatures(1000).
  setInputCol(tokenizer.getOutputCol).
  setOutputCol("words_hach")

val idf = new IDF().
  setInputCol(hashingTF.getOutputCol).
  setOutputCol("features")

// Logistic regression
val lr = new LogisticRegression().
  setMaxIter(100).
  setRegParam(0.01)

// Create our pipeline
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, lr))

// Train the model
val model = pipeline.fit(training_df)

// Predict on the test dataset
val test_pred = model.transform(test_df) //.select('text, 'label, 'prediction)

// Error rates :
//general
//test_pred.select('label,'prediction).map( l => l(0)==l(1)).groupBy(l=>l).map(l=>(l._1,l._2.size)).take(5)
//positive
//test_pred.select('label,'prediction).rdd.filter(l=>(l(0)==0)).map( l => l(0)==l(1)).groupBy(l=>l).map(l=>(l._1,l._2.size)).take(5)
//négative
//test_pred.select('label,'prediction).rdd.filter(l=>(l(0)==1)).map( l => l(0)==l(1)).groupBy(l=>l).map(l=>(l._1,l._2.size)).take(5)

// Saving the model
model.save(path + "my_pipeline")

// Load the model
val model_save = PipelineModel.load(path + "my_pipeline")

// Apply on test dataset (again)
val test_pred_save = model_save.transform(test_df) //.select('text, 'label, 'prediction)
//test_pred_save.select('label,'prediction).map( l => l(0)==l(1)).groupBy(l=>l).map(l=>(l._1,l._2.size)).take(3)

// Some examples 
val exemple = sc.parallelize( List( 
  ("je suis content",1.0) ,
  ("c'est nul", 0.0) ,
  ("Une merveille. Un coup d'éclat, un feu d'artifice. Moi qui suis rebutée par toute comédie musicale, j'ai été happée. De loin la plus belle adaptation de l'oeuvre du grand Victor. Les acteurs sont tous magnifiques (ma palme à Russell Crowe et à l'adorable Amanda Seyfried). Un grand film magique et tragique. Inoubliable !" , 1.0 ),
  ("Quel intérêt de faire chanter absolument TOUS les dialogues ? Au bout d'une heure j'en avais marre le rythme en est donc devenu catastrophique, ça s'essouffle rapidement et on décroche il aurait fallu faire des pauses avec des dialogues parlés, entrecoupés de chansons.", 0.0)
) ).toDF("text","label")

val pred_exemple = model_save.transform(exemple)
//pred_exemple.select('label,'prediction).take(10)
//.map( l => l(0)==l(1)).groupBy(l=>l).map(l=>(l._1,l._2.size)).take(5)

