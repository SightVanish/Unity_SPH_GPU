using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Diagnostics;
using System.IO;
using System.Text;
using System;
using System.Runtime.Serialization.Formatters.Binary;

struct MyData
{
    public float[] myData;
}

public class Inference : MonoBehaviour
{
    float[] input_data;
    float[] output_data;


    private void Awake()
    {

    }

    private void WriteData()
    {
        input_data = new float[14];

        for (int i = 0; i < input_data.Length; i++)
        {
            input_data[i] = 0.12f * (i+1);
        }

        if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\input.bin"))
        {
            Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\input.bin");
        }

        FileStream saveFile = new FileStream(@"F:\UnityGames\SPHGPU\dataset\input.bin", FileMode.Create);
        saveFile.Close();

        FileStream saveFile1 = new FileStream(@"F:\UnityGames\SPHGPU\dataset\input.bin", FileMode.Append);
        var writer1 = new BinaryWriter(saveFile1);

        foreach (float f in input_data)
        {
            writer1.Write(f);
        }
        saveFile1.Flush();
        saveFile1.Close();

        ReadBin();
    }

    private void ReadBin()
    {

        using var readFile = File.OpenRead(@"F:\UnityGames\SPHGPU\dataset\output.bin");
        using var reader = new BinaryReader(readFile);

        int nFloats = (int)readFile.Length / sizeof(float);
        float[] input = new float[nFloats];

        for (int i = 0; i < nFloats; ++i)
        {
            input[i] = reader.ReadSingle();
            print(input[i]);
        }

        readFile.Close();


        UnityEditor.EditorApplication.isPlaying = false;
    }

}