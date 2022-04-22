using UnityEngine;
//using System.Net;
using System.Net.Sockets;
using System;

public class Inference : MonoBehaviour
{
    public string ip = "127.0.0.1";
    public int port = 60000;
    private Socket client;
    [SerializeField]
    private float[] dataOut, dataIn; //debugging


    private void SendData(float[] dataOut)
    {
        client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        client.Connect(ip, port);

        var byteArray = new byte[dataOut.Length * 4];
        Buffer.BlockCopy(dataOut, 0, byteArray, 0, byteArray.Length);
        client.Send(byteArray);

        client.Close();
    }
    private float[] ReceiveData()
    {
        float[] floatsReceived;
        client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        client.Connect(ip, port);
        
        //allocate and receive bytes
        byte[] bytes = new byte[4000];
        int idxUsedBytes = client.Receive(bytes);

        //convert bytes to floats
        floatsReceived = new float[idxUsedBytes / 4];
        Buffer.BlockCopy(bytes, 0, floatsReceived, 0, idxUsedBytes);

        client.Close();
        return floatsReceived;
    }


    private void Start()
    {
        dataOut = new float[] { 1.0f, 2.0f, 3.0f };
    }

    private void Update()
    {
        SendData(dataOut);
        
        dataIn = ReceiveData();
        foreach (var i in dataIn)
            print(i);
        
    }
}