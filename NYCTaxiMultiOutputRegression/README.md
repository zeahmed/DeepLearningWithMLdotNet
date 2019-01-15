# Multi-Output Regression with ML.Net and TensorFlow

This sample describes how to create a multi-output regression model using ML.Net and TensorFLow. Multi-output regression, also known as multi-target, multi-variate, or multi-response  regression, aims to simultaneously predict multiple real-valued output/target variables.

In ML.Net, this regression task can be modelled using ML.Net's TensorFlow scoring and training component. 

## TensorFlow Scoring in ML.Net
ML.Net has components called [TensorFLowTransformer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.tensorflowtransform?view=ml-dotnet) and [TensorFlowEstimator](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.tensorflowestimator?view=ml-dotnet) that can be used for 

* Scoring with pretrained TensorFlow model where the `TensorFLowTransformer` extracts hidden layers' values from a pre-trained Tensorflow model and uses outputs as features in ML.Net pipeline. 

* Retraining of TensorFlow model where the `TensorFlowEstimator` retrains a TensorFlow model using the user data passed through ML.Net pipeline. Once the model is trained, it's outputs can be used as features for scoring. 

## Multi-Output Regression Model

Let's create a multi-output regression model in TensorFlow and use it in ML.Net. The following script creates the TensorFLow graph. Note that this script only creates the graph and does not do any training. The actual training will be done in ML.Net script.

```python
import tensorflow as tf

f_size = 15 # Number of features passed from ML.Net
num_output = 2 # Number of outputs
tf.set_random_seed(1)
X = tf.placeholder('float', [None, f_size], name="X")
Y = tf.placeholder('float', [None, num_output], name="Y")
lr = tf.placeholder(tf.float32, name = "learning_rate")


# Set model weights
W = tf.Variable(tf.random_normal([f_size,num_output], stddev=0.1), name = 'W')
b = tf.Variable(tf.zeros([num_output]), name = 'b')

l1 = 0
l2 = 0
RegScores = tf.add(tf.matmul(X, W), b, name='RegScores')
loss = tf.reduce_mean(tf.square(Y-tf.squeeze(RegScores))) / 2  + l2 * tf.nn.l2_loss(W) + l1 * tf.reduce_sum(tf.abs(W))
loss = tf.identity(loss, name="Loss")
optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, name='MomentumOptimizer').minimize(loss)

init = tf.global_variables_initializer()
# Launch the graph.
with tf.Session() as sess:
    sess.run(init)
    tf.saved_model.simple_save(sess, r'NYCTaxi/model', inputs={'X': X, 'Y': Y}, outputs={'RegScores': RegScores} )
```

Here, 

* `X` and `Y` are the input feature vector and label placeholders.
* `W` and `b` are the parameters of the model.
* `RegScores` is the predicted value. It is vector of length 2 in this case.
* `learning_rate` is placeholder that is dynamically set during training from ML.Net
* `MomentumOptimizer` is the name of optimization operator in the graph that will be used for training.

 Upon executing, this python script will create a checkpoint model directory called `NYCTaxi/model`.

 ## Training and Prediction with Multi-Output LR model in ML.Net

 The sample uses the dataset from [ML.Net's TaxiFarePrediction](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/Regression_TaxiFarePrediction) getting-started tutorial. The tutorial explains how to use ML.Net for predicting single real-valued target variable i.e. predicting the `FareAmount`. However, this sample tries to predict `FareAmount` together with `TipAmount`.

### 1. Loading Data
This sample loads 8 columns from the dataset instead of 7 in [ML.Net's TaxiFarePrediction](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/Regression_TaxiFarePrediction) getting-started tutorial. The additional columns `TipAmount` is used as second target variable.
 ``` csharp
 string TrainDataPath = "NYCTaxi/train.csv";
string TestDataPath = "NYCTaxi/test.csv";

//Create ML Context with seed for repeteable/deterministic results
MLContext mlContext = new MLContext(seed: 0);

// STEP 1: Common data loading configuration
TextLoader textLoader = mlContext.Data.CreateTextReader(new[]
                                                        {
                                                            new TextLoader.Column("VendorId", DataKind.Text, 0),
                                                            new TextLoader.Column("RateCode", DataKind.Text, 1),
                                                            new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                                                            new TextLoader.Column("TripTime", DataKind.R4, 3),
                                                            new TextLoader.Column("TripDistance", DataKind.R4, 4),
                                                            new TextLoader.Column("PaymentType", DataKind.Text, 5),
                                                            new TextLoader.Column("FareAmount", DataKind.R4, 6),
                                                            new TextLoader.Column("TipAmount", DataKind.R4, 7)
                                                        },
                                                            hasHeader: true,
                                                            separatorChar: ','
                                                        );

IDataView baseTrainingDataView = textLoader.Read(TrainDataPath);
IDataView testDataView = textLoader.Read(TestDataPath);
 ```

 ### 2. Building the Pipeline
The major difference here is in building the pipeline where `TensorFlowEstimator` is used to estimate the model parameters created with TensorFlow script above.
 ``` csharp
var dataProcessPipeline = mlContext.Transforms.Concatenate("Y", "FareAmount", "TipAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId", "VendorIdEncoded"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode", "RateCodeEncoded"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType", "PaymentTypeEncoded"))
                .Append(mlContext.Transforms.Normalize(inputName: "PassengerCount", mode: NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "TripTime", mode: NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "TripDistance", mode: NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Concatenate("X", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", "PassengerCount", "TripTime", "TripDistance"))
                .Append(new TensorFlowEstimator(mlContext, new TensorFlowTransform.Arguments()
                {
                    ModelLocation = "NYCTaxi/model", // Model is created with this script: DeepLearningWithMLdotNet\NYCTaxiMultiOutputRegression\TF_MultiOutputLR.py
                    InputColumns = new[] { "X" },
                    OutputColumns = new[] { "RegScores" },
                    LabelColumn = "Y",
                    TensorFlowLabel = "Y",
                    OptimizationOperation = "MomentumOptimizer",
                    LossOperation = "Loss",
                    Epoch = 10,
                    LearningRateOperation = "learning_rate",
                    LearningRate = 0.01f,
                    BatchSize = 20,
                    ReTrain = true
                }));
```

In this ML.Net pipeline, the `FareAmount` and `TipAmount` are combined into a vector-valued column called `Y`. `TensorFlowEstimator` uses the `X` (input), `RegScores` (output), `Y` (Label) and optimization related operator names (`MomentumOptimizer`, `Loss`, `learning_rate` etc.) for retraining of model created with TensorFlow script above.

### 3. Predicting with Trained Model
Once the model is trained with 
```csharp
var trainedModel = dataProcessPipeline.Fit(baseTrainingDataView);
```
, it can be used for prediction as any other ML.Net model.
```csharp
// Create prediction function and test prediction
var predictFunction = trainedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

var oneSample = new TaxiTrip()
{
    VendorId = "CMT",
    RateCode = "1",
    PassengerCount = 2,
    TripTime = 1405,
    TripDistance = 10.3f,
    PaymentType = "CRD",
    FareAmount = 0, // To predict. Actual/Observed = 31.0
    TipAmount = 0 // To predict. Actual/Observed = 7.36
};

var prediction = predictFunction.Predict(oneSample);
Console.WriteLine("[FareAmount, TipAmount] = [{0}]", string.Join(", ", prediction.RegScores));
```
Note that, the `RegScores` (the output) is vector type instead of scalar type as used in [ML.Net's TaxiFarePrediction](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/Regression_TaxiFarePrediction) getting-started tutorial.
```csharp
public class TaxiTripFarePrediction
{
    [VectorType(2)]
    public float[] RegScores;  // This is vector because its a MultiOuput regression problem.
}
```

### 4. Evaluating Model

Currently, evaluation (i.e. computing metrics such as Root Mean Square Error (RMSE), log-loss etc.) on the test data does not work because ML.Net's regression evaluator does not work on vector-valued.
```csharp
// The evaluation does not work. It requires the score to be scalar.
// However, for multi-output regression its vector
// var predicted = trainedModel.Transform(testDataView);
// var metrics = mlContext.Regression.Evaluate(predicted, "Y", "RegScores");
```