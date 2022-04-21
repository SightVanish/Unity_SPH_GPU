//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//using Unity.Barracuda;
//using System.Diagnostics;
//using System.IO;
//using System.Text;
//using System;
//using System.Runtime.Serialization.Formatters.Binary;

//struct MyData
//{
//    public float[] myData;
//}

//public class Inference : MonoBehaviour
//{
//    float[] input_data;
//    float[] output_data;


//    private void Awake()
//    {
//        WriteData();
//        foreach (var i in input_data)
//            print(i);

//        RunPython();

//        // ReadBin();
//        // foreach (var i in output_data)
//        //     print(i);
//    }

//    private void RunPython()
//    {
//        Process psi = new Process();

//        string script = @"F:\UnityGames\SPHGPU\Pytorch_scripts\inference.py";

//        psi.StartInfo.FileName = @"C:\Users\11054\anaconda3\envs\SPH\python.exe";

//        psi.StartInfo.UseShellExecute = false;
//        psi.StartInfo.Arguments = script;
//        psi.StartInfo.RedirectStandardOutput = true;
//        psi.StartInfo.RedirectStandardError = true;
//        psi.StartInfo.RedirectStandardInput = true;

//        // psi.StartInfo.CreateNoWindow = true;

//        psi.Start();
//        psi.BeginOutputReadLine();
//        psi.OutputDataReceived += new DataReceivedEventHandler(get_output);
//        psi.WaitForExit();
//    }
//    private void get_output(object sender, DataReceivedEventArgs e)
//    {
//        print(e.Data);
//    }
//    private void WriteData()
//    {
//        input_data = new float[5];

//        for (int i = 0; i < input_data.Length; i++)
//        {
//            input_data[i] = 0.12f * (i+1);
//        }

//        if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\input.bin"))
//        {
//            Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\input.bin");
//        }

//        FileStream saveFile = new FileStream(@"F:\UnityGames\SPHGPU\dataset\input.bin", FileMode.Create);
//        var writer = new BinaryWriter(saveFile);

//        foreach (float f in input_data)
//        {
//            writer.Write(f);
//        }
//        saveFile.Flush();
//        saveFile.Close();
//    }

//    private void ReadBin()
//    {

//        using var readFile = File.OpenRead(@"F:\UnityGames\SPHGPU\dataset\output.bin");
//        using var reader = new BinaryReader(readFile);

//        int nFloats = (int)readFile.Length / sizeof(float);
//        float[] input = new float[nFloats];

//        for (int i = 0; i < nFloats; ++i)
//        {
//            input[i] = reader.ReadSingle();
//            print(input[i]);
//        }

//        readFile.Close();

//    }

//}

using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;
using System;

public class Inference : MonoBehaviour
{
    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    Vector3 receivedPos = Vector3.zero;
`
    bool running;

    private void Update()
    {
        transform.position = receivedPos;
    }

    private void Start()
    {
        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();
    }

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();
        
        running = true;
        SendThenReceiveData();

        while (running)
        {
            // SendAndReceiveData();
            // SendThenReceiveData();

        }
        listener.Stop();
    }

    void SendThenReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        

        var floatArray1 = new float[] { 123.45f, 123f, 45f, 1.2f, 34.5f };

        // send data
        byte[] byteArray = new byte[floatArray1.Length * 4];
        Buffer.BlockCopy(floatArray1, 0, byteArray, 0, byteArray.Length);
        nwStream.Write(byteArray, 0, byteArray.Length);

        // receive data
        byte[] buffer = new byte[client.ReceiveBufferSize]; // data received
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);

        print(buffer[0]);
        floatArray1[0] = BitConverter.ToSingle(buffer, 0);
        floatArray1[1] = BitConverter.ToSingle(buffer, 4);
        floatArray1[2] = BitConverter.ToSingle(buffer, 8);
        floatArray1[3] = BitConverter.ToSingle(buffer, 12);
        floatArray1[4] = BitConverter.ToSingle(buffer, 16);
        print(floatArray1[0]);
        print(floatArray1[1]);
        print(floatArray1[2]);
        print(floatArray1[3]);
        print(floatArray1[]);

    }
    void SendAndReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize]; // data received

        //---receiving Data from the Host----
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize); //Getting data in Bytes from Python
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead); //Converting byte data to string

        if (dataReceived != null)
        {
            //---Using received data---
            receivedPos = StringToVector3(dataReceived); //<-- assigning receivedPos value from Python
            print("received pos data, and moved the Cube!");

            //---Sending Data to Host----
            byte[] myWriteBuffer = Encoding.ASCII.GetBytes("Hey I got your message Python! Do You see this massage?"); //Converting string to byte data
            nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); //Sending the data in Bytes to Python
        }
    }

    public static Vector3 StringToVector3(string sVector)
    {
        // Remove the parentheses
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        // split the items
        string[] sArray = sVector.Split(',');

        // store as a Vector3
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }

}