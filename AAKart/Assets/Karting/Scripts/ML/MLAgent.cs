using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;


public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers-1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }


    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }

    public List<float[,]> GetCoefficients()
    {
        return coeficients;
    }

    public List<float[]> GetIntercepts()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }

    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(Perception p, Transform transform, bool usingPosition)
    {
        Parameters parameters = Record.ReadParameters(8, Time.timeSinceLevelLoad, p, transform);
        float[] input = parameters.ConvertToFloatArrat();
        //Debug.Log("input " + input.Length);

        // Ignorar parametros que no nos interesa
        float[] cleanInput;
        if (usingPosition) cleanInput= new float[7];
        else cleanInput= new float[5];

        int cleanInputIndex = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (usingPosition)
            {
                if (!(i == 6 || i == 8))
                {
                    cleanInput[cleanInputIndex] = input[i];
                    cleanInputIndex++;
                }
            }
            else
            {
                if (!(i == 5 || i == 6 || i == 7 || i == 8))
                {
                    cleanInput[cleanInputIndex] = input[i];
                    cleanInputIndex++;
                }
            }
        }

        return FeedForwardLogic(NormalizeInput(cleanInput));
    }

    //Normalizamos el input, para que sean valores parecidos a los con los que se entrenó
    float[] NormalizeInput(float[] input)
    {
        float[] normalized = new float[input.Length];

        // Calcular la media y la desviación estándar para cada columna
        float mean = input.Sum() / input.Length;

        float sumOfSquaresOfDifferences = input.Select(val => Mathf.Pow((val - mean), 2)).Sum();
        float stdDev = Mathf.Sqrt(sumOfSquaresOfDifferences / input.Length);

        for (int i = 0; i < input.Length; i++)
        {
            normalized[i] = (input[i] - mean) / stdDev;
        }

        return normalized;
    }


    public float[] FeedForwardLogic(float[] inputs)
    {
        List<float[,]> coefficients = mlpParameters.GetCoefficients();
        List<float[]> intercepts = mlpParameters.GetIntercepts();

        if (inputs.Length != coefficients[0].GetLength(1))
            return new float[coefficients[coefficients.Count-1].GetLength(0)];


        float[] currentLayerOutput = inputs;

        // Loop a través de las capas
        for (int layerIndex = 0; layerIndex < coefficients.Count; layerIndex++)
        {
            float[] layerOutput = new float[coefficients[layerIndex].GetLength(0)];

            // Loop a través de las neuronas de la capa actual
            for (int neuronIndex = 0; neuronIndex < coefficients[layerIndex].GetLength(0); neuronIndex++)
            {
                float weightedSum = intercepts[layerIndex][neuronIndex]; // Sesgo

                // Loop a través de las entradas y sus pesos
                for (int inputIndex = 0; inputIndex < coefficients[layerIndex].GetLength(1); inputIndex++)
                {
                    weightedSum += currentLayerOutput[inputIndex] * coefficients[layerIndex][neuronIndex, inputIndex];
                }

                // Aplicar función de activación
                layerOutput[neuronIndex] = ActivationFunction(weightedSum);
            }

            currentLayerOutput = layerOutput; // La salida de esta capa se convierte en la entrada para la siguiente
        }

        return currentLayerOutput; // La salida final después de todas las capas
    }

    // En este caso la sigmoidal
    private float ActivationFunction(float x)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }


    /// <summary>
    /// Implements the conversion of the output value to the action label. 
    /// Depending on what actions you have chosen or saved in the dataset, and in what order, the way it is converted will be one or the other.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public Labels ConvertIndexToLabel(int index)
    {
        //TODO: implement the conversion from index to actions.

        switch (index)
        {
            case 0:
                return Labels.ACCELERATE;

            case 1:
                return Labels.BRAKE;

            case 2:
                return Labels.LEFT_ACCELERATE;

            case 3:
                return Labels.LEFT_BRAKE;

            case 4:
                return Labels.NONE;

            case 5:
                return Labels.RIGHT_ACCELERATE;

            case 6:
                return Labels.RIGHT_BRAKE;

            default:
                return Labels.NONE;
        }
    }

    public Labels Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        Labels label = ConvertIndexToLabel(index);
        return label;
    }

    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        max = output[0];
        int index = 0;
        for(int i = 1; i < output.Length; i++)
        {
            if(output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}

public class MLAgent : MonoBehaviour
{
    public enum ModelType { MLP=0, KNN=1 }
    public ModelType model;
    public bool usingPositions;
    public TextAsset MLPText;
    public TextAsset KNNText;
    public bool agentEnable;

    private MLPParameters mlpParameters;
    private MLPModel mlpModel;

    private KNNModel kNNModel;

    private Perception perception;

    // Start is called before the first frame update
    void Start()
    {
        if (agentEnable)
        {
            string file = MLPText.text;
            if (model == ModelType.MLP)
            {
                mlpParameters = LoadParameters(MLPText.text);
                mlpModel = new MLPModel(mlpParameters);
            }
            else if (model == ModelType.KNN)
            {
                kNNModel = new KNNModel();
                kNNModel.LoadParameters(KNNText.text, usingPositions);
            }
            perception = GetComponent<Perception>();
        }
    }



    public KartGame.KartSystems.InputData AgentInput()
    {
        Labels label = Labels.NONE;
        switch (model)
        {
            case ModelType.MLP:
                float[] outputs = this.mlpModel.FeedForward(perception,this.transform, usingPositions);
                label = this.mlpModel.Predict(outputs);
                break;
            case ModelType.KNN:
                label = kNNModel.Predict(perception, this.transform);
                break;
        }
        KartGame.KartSystems.InputData input = Record.ConvertLabelToInput(label);

        return input;
    }

    public static string TrimpBrackers(string val)
    {
        val = val.Trim();
        val = val.Substring(1);
        val = val.Substring(0, val.Length - 1);
        return val;
    }

    public static int[] SplitWithColumInt(string val)
    {
        val = val.Trim();
        string[] values =val.Split(",");
        int[] result = new int[values.Length];
        for(int i = 0; i < values.Length; i++)
        {
            values[i] = values[i].Trim();
            if (values[i].StartsWith("'"))
                values[i] = values[i].Substring(1);
            if (values[i].EndsWith("'"))
                values[i] = values[i].Substring(0, values[i].Length-1);
            result[i] = int.Parse(values[i]);
        }
        return result;
    }

    public static float[] SplitWithColumFloat(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = float.Parse(values[i], System.Globalization.CultureInfo.InvariantCulture);
        }
        return result;
    }

    public static MLPParameters LoadParameters(string file)
    {
        string[] lines = file.Split("\n");
        int num_layers = 0;
        MLPParameters mlpParameters = null;
        int currentParameter = -1;
        int[] currentDimension = null;
        bool coefficient = false;
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            line = line.Trim();
            if(line != "")
            {
                string[] nameValue = line.Split(":");
                string name = nameValue[0];
                string val = nameValue[1];
                if (name == "num_layers")
                {
                    num_layers = int.Parse(val);
                    mlpParameters = new MLPParameters(num_layers);
                }
                else
                {
                    if (num_layers <= 0)
                        Debug.LogError("Format error: First line must be num_layers");
                    else
                    {
                        if (name == "parameter")
                            currentParameter = int.Parse(val);
                        else if (name == "dims")
                        {
                            val = TrimpBrackers(val);
                            currentDimension = SplitWithColumInt(val);
                        }
                        else if (name == "name")
                        {
                            if (val.StartsWith("coefficient"))
                            {
                                coefficient = true;
                                int index = currentParameter / 2;
                                mlpParameters.CreateCoeficient(currentParameter, currentDimension[0], currentDimension[1]);
                            }
                            else
                            {
                                coefficient = false;
                                mlpParameters.CreateIntercept(currentParameter, currentDimension[0]);
                            }

                        }
                        else if (name == "values")
                        {
                            val = TrimpBrackers(val);
                            float[] parameters = SplitWithColumFloat(val);

                            for (int index = 0; index < parameters.Length; index++)
                            {
                                if (coefficient)
                                {
                                    int row = index / currentDimension[1];
                                    int col = index % currentDimension[1];
                                    mlpParameters.SetCoeficiente(currentParameter, row, col, parameters[index]);
                                }
                                else
                                {
                                    mlpParameters.SetIntercept(currentParameter, index, parameters[index]);
                                }
                            }
                        }
                    }
                }
            }
        }
        return mlpParameters;
    }
}
