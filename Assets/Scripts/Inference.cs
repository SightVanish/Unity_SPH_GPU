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

    private void Start()
    {
        input_data = new float[12];

        LoadFromBin();

        for (int i = 0; i < input_data.Length; i++)
        {
            // input_data[i] = 0.35f * (i+1);
            print(input_data[i]);
        }

        // Save2Bin();
    }

    private void Save2Bin()
    {
        if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\input.bin"))
        {
            Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\input.bin");
        }

        float[] data = new float[10];
        using FileStream saveFile = File.Create(@"d:\tmp\test.bin");

        foreach (float f in data)
        {
            saveFile.Write(BitConverter.GetBytes(f), 0, sizeof(float));
        }

        saveFile.Flush();
        saveFile.Close();
    }
    private void LoadFromBin()
    {
        BinaryFormatter formatter = new BinaryFormatter();
        FileStream saveFile = File.Open(@"F:\UnityGames\SPHGPU\dataset\input.bin", FileMode.Open);
        input_data = (float[])formatter.Deserialize(saveFile);
        saveFile.Close();
    }

    private void CSVWrite()
    {
        input_data = new float[10];
        output_data = new float[10];

        for (int i = 0; i < input_data.Length; i++)
            input_data[i] = 1.5f + i;

        var psi = new Process();
        psi.StartInfo.FileName = @"C:\Users\11054\anaconda3\envs\SPH\python.exe";

        string script = @"F:\UnityGames\SPHGPU\Pytorch_scripts\inference.py";

        psi.StartInfo.UseShellExecute = false;
        psi.StartInfo.CreateNoWindow = true;
        psi.StartInfo.Arguments = script;
        psi.StartInfo.RedirectStandardError = true;
        psi.StartInfo.RedirectStandardOutput = true;
        psi.StartInfo.RedirectStandardInput = true;

        psi.Start();
        psi.BeginOutputReadLine();
        psi.WaitForExit();

        WriteDataToCSV(@"F:\UnityGames\SPHGPU\dataset\input.csv");

        ReadCSVToData(@"F:\UnityGames\SPHGPU\dataset\output.csv");

        foreach (var i in output_data)
            print(i);
        UnityEditor.EditorApplication.isPlaying = false;
        

    }

    public void WriteDataToCSV(string path)
    {

        if (!File.Exists(path))
            File.Create(path).Close();

        StreamWriter sw = new StreamWriter(path, true, Encoding.UTF8);
        for (int i = 0; i < input_data.Length; i++)
        {
            sw.Write(input_data[i] + ",");
        }

        sw.Write("\n");


        sw.Flush();
        sw.Close();
    }

    public void ReadCSVToData(string path)
    {

        string[] lines = File.ReadAllLines(path);
        int i = 0;
        foreach (string line in lines)
        {
            string[] columns = line.Split(',');
            foreach (string column in columns)
            {
                output_data[i] = Convert.ToSingle(column);
                i++;
            }
        }
    }


}