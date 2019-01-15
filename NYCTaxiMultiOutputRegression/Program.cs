using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Transforms.Normalizers.NormalizingEstimator;

namespace NYCTaxiMultiOutputRegression
{
    class Program
    {
        public class TaxiTrip
        {
            public string VendorId;
            public string RateCode;
            public float PassengerCount;
            public float TripTime;
            public float TripDistance;
            public string PaymentType;
            public float FareAmount;
            public float TipAmount;
        }

        public class TaxiTripFarePrediction
        {
            [VectorType(2)]
            public float[] RegScores;  // This is vector because its a MultiOuput regression problem.
        }

        static void Main(string[] args)
        {

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

            //Sample code of removing extreme data like "outliers" for FareAmounts higher than $150 and lower than $1 which can be error-data 
            var cnt = baseTrainingDataView.GetColumn<float>(mlContext, "FareAmount").Count();
            IDataView trainingDataView = mlContext.Data.FilterByColumn(baseTrainingDataView, "FareAmount", lowerBound: 1, upperBound: 150);
            var cnt2 = trainingDataView.GetColumn<float>(mlContext, "FareAmount").Count();

            // STEP 2: Common data process configuration with pipeline data transformations
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

            var trainedModel = dataProcessPipeline.Fit(baseTrainingDataView);


            // The evaluation does not work. It requires the score to be scalar.
            // However, for multi-output regression its vector
            // var predicted = trainedModel.Transform(testDataView);
            // var metrics = mlContext.Regression.Evaluate(predicted, "Y", "RegScores");

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
            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }
    }
}
