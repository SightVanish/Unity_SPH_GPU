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
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    private Thread mThread;
    private IPAddress localAdd;
    private TcpListener listener;
    private TcpClient client;

    private float[] sendData;
    private float[] receivedData;
    
    private void Update()
    {
        sendData = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        if (receivedData != null)
        {
            print(receivedData[0]);
            // print(floatArray1[1]);
            // print(floatArray1[2]);
            // print(floatArray1[3]);
            // print(floatArray1[4]);
        }
        receivedData = null;
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
        
        while(true)
        {
            SendThenReceiveData();
        }

        listener.Stop();
    }

    void SendThenReceiveData()
    {
        NetworkStream nwStream = client.GetStream();

        // send data
        if (sendData != null)
        {
            byte[] byteArray = new byte[sendData.Length * 4];
            Buffer.BlockCopy(sendData, 0, byteArray, 0, byteArray.Length);
            nwStream.Write(byteArray, 0, byteArray.Length);

            sendData = null;
        }

        if (receivedData == null)
        {
            receivedData = new float[5];

            // receive data
            byte[] buffer = new byte[client.ReceiveBufferSize];
            int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);

            receivedData[0] = BitConverter.ToSingle(buffer, 0);
            receivedData[1] = BitConverter.ToSingle(buffer, 4);
            receivedData[2] = BitConverter.ToSingle(buffer, 8);
            receivedData[3] = BitConverter.ToSingle(buffer, 12);
            receivedData[4] = BitConverter.ToSingle(buffer, 16);
        }

    }

}