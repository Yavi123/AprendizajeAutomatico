using log4net.Util;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Security.Cryptography;
using UnityEngine;

public class KNNModel
{

    float[][] data = null;

    bool usingPosition;

    // Start is called before the first frame update
    public void LoadParameters(string file, bool usingPosition)
    {
        this.usingPosition = usingPosition;

        string[] lines = file.Split("\n");
        data = new float[lines.Length][];

        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            line = line.Trim();
            if (line != "")
            {
                data[i] = new float[usingPosition ? 8 : 6];
                string[] nameValue = line.Split(",");
                for (int j = 0; j < 5; j++)
                {
                    data[i][j] = float.Parse(nameValue[j], System.Globalization.CultureInfo.InvariantCulture);

                    //Como un valor negativo significa no hay nada cerca, KNN se puede confundir, creyendo que si esta a 0 de una pared, es parecido a no tener nada
                    if (data[i][j] < -0.1f)
                    {
                        data[i][j] = 100f;
                    }
                }

                if (usingPosition)
                {
                    for (int j = 5; j < 7; j++)
                    {
                        data[i][j] = float.Parse(nameValue[j == 5 ? 5 : 7], System.Globalization.CultureInfo.InvariantCulture);
                    }
                }

                switch (nameValue[nameValue.Length - 1])
                {
                    case "ACCELERATE":
                        data[i][data[0].Length - 1] = 0;
                        break;
                    case "BRAKE":
                        data[i][data[0].Length - 1] = 1;
                        break;
                    case "LEFT_ACCELERATE":
                        data[i][data[0].Length - 1] = 2;
                        break;
                    case "LEFT_BRAKE":
                        data[i][data[0].Length - 1] = 3;
                        break;
                    case "NONE":
                        data[i][data[0].Length - 1] = 4;
                        break;
                    case "RIGHT_ACCELERATE":
                        data[i][data[0].Length - 1] = 5;
                        break;
                    case "RIGHT_BRAKE":
                        data[i][data[0].Length - 1] = 6;
                        break;
                    default:
                        data[i][data[0].Length - 1] = 1;
                        break;
                }
            }
        }
    }

    internal Labels Predict(Perception perception, UnityEngine.Transform transform)
    {
        float[] input = CleanInput(perception, transform);

        float[] mostSimilar = new float[data[0].Length];
        float difference = float.MaxValue;

        for (int i = 0; i < data.Length; i++)
        {
            float currentDiff = Euclidean(data[i], input);

            if (currentDiff < difference)
            {
                mostSimilar = data[i];
                difference = currentDiff;
            }

        }

        return IndexToLabel(mostSimilar[mostSimilar.Length - 1]);
    }

    float Euclidean(float[] a, float[] b)
    {
        int lesserLenght = Mathf.Min(a.Length, b.Length);
        float sum = 0;
        for (int i = 0; i < lesserLenght; i++)
        {
            sum += Mathf.Pow(a[i] - b[i], 2);
        }
        return Mathf.Sqrt(sum);
    }

    Labels IndexToLabel(float index)
    {
        if (index < 0.5f)
            return Labels.ACCELERATE;
        else if (index < 1.5f)
            return Labels.BRAKE;
        else if (index < 2.5f)
            return Labels.LEFT_ACCELERATE;
        else if (index < 3.5f)
            return Labels.LEFT_BRAKE;
        else if (index < 4.5f)
            return Labels.NONE;
        else if (index < 5.5f)
            return Labels.RIGHT_ACCELERATE;
        else if (index < 6.5f)
            return Labels.RIGHT_BRAKE;
        else
            return Labels.NONE;
    }

    float[] CleanInput(Perception p, UnityEngine.Transform transform)
    {
        Parameters parameters = Record.ReadParameters(8, Time.timeSinceLevelLoad, p, transform);
        float[] input = parameters.ConvertToFloatArrat();

        // Ignorar parametros que no nos interesa
        float[] cleanInput;
        if (usingPosition) cleanInput = new float[7];
        else cleanInput = new float[5];

        int cleanInputIndex = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (usingPosition)
            {
                if (!(i == 6 || i == 8))
                {
                    if (i < 5 && input[i] < -0.1f) input[i] = 100f;
                    cleanInput[cleanInputIndex] = input[i];
                    cleanInputIndex++;
                }
            }
            else
            {
                if (!(i == 5 || i == 6 || i == 7 || i == 8))
                {
                    if (i < 5 && input[i] < -0.1f) input[i] = 100f;
                    cleanInput[cleanInputIndex] = input[i];
                    cleanInputIndex++;
                }
            }
        }

        return cleanInput;
    }
}
