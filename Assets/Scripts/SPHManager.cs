using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using System;
using System.IO;
using System.Text;
using UnityEditor;
using Random = UnityEngine.Random;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Linq;

struct Particle
{
    public Vector3 position;
    public Vector4 colorGradient;
}

public class SPHManager : MonoBehaviour
{
    [Header("==== Compute Shader ====")]
    public ComputeShader computeShaderSPH;

    [Header("==== Particles Rendering ====")]
    public float radius = 2f;
    public Mesh particleMesh;
    public float particleRenderSize = 1f;
    public Material particleMaterial;

    public int numberOfParticles = 2000;
    public int dimensions = 100;
    public int particlePerRow = 50;
    public float initParticleDensity = 1.5f;

    [Header("==== Particle Properties ====")]
    public float stiffness = 0.2f;
    public float mass = 1f;
    public float gasConstant = 2000f;
    public float restDensity = 2f;
    public float viscosityCoefficient = 1.2f;


    [Header("==== Rendering Properties")]
    public float damping = -0.5f;
    public float dt = 0.01f;


    [Header("==== Neighbours ====")]
    public int maximumParticlesPerCell = 100;

    // pre-computed value
    private float[] gravity = { 0.0f, -9.81f, 0.0f };
    private float radius2;
    private float radius3;
    private float radius4;
    private float radius5;
    private float mass2;


    private int[] _neighbourTracker;
    private Particle[] _particles;
    private int[] _neighbourList;
    private uint[] _hashGrid;
    private float[] _densities;
    private float[] _pressures;
    private Vector3[] _velocities;
    private Vector3[] _forces;

    // buffers
    private ComputeBuffer _particlesBuffer;
    private ComputeBuffer _argsBuffer;
    private ComputeBuffer _neighbourListBuffer;
    private ComputeBuffer _neighbourTrackerBuffer;
    private ComputeBuffer _hashGridBuffer;
    private ComputeBuffer _hashGridTrackerBuffer;
    private ComputeBuffer _densitiesBuffer;
    private ComputeBuffer _pressuresBuffer;
    private ComputeBuffer _velocitiesBuffer;
    private ComputeBuffer _forcesBuffer;

    // for material
    private static readonly int SizeProperty = Shader.PropertyToID("_size");
    private static readonly int ParticlesBufferProperty = Shader.PropertyToID("_particlesBuffer");

    // kernel
    private int clearHashGridKernel;
    private int recalculateHashGridKernel;
    private int buildNeighbourListKernel;
    private int computeDensityPressureKernel;
    private int computeForcesKernel;
    private int integrateKernel;

    [Header("==== Collection Data ====")]
    public bool save_data = true;
    public bool delete_data = true;
    public int save_interval = 5;
    private int counter = 0;
    private int save_counter = 0;

    [Header("==== Inference ====")]
    public bool inference = true;
    public bool use_python = false;
    public int init_interval = 20;
    private int init_counter = 0;
    public int inference_interval = 5;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    private Thread mThread;
    private IPAddress localAdd;
    private TcpListener listener;
    private TcpClient client;

    public string ip = "127.0.0.1";
    public int port = 60000;
    private Socket socketClient;



    private void Start()
    {
        Application.targetFrameRate = 30;

        radius2 = Mathf.Pow(radius, 2);
        radius3 = Mathf.Pow(radius, 3);
        radius4 = Mathf.Pow(radius, 4);
        radius5 = Mathf.Pow(radius, 5);
        mass2 = Mathf.Pow(mass, 2);

        RespawnParticles();
        FindKernels();
        InitComputeBuffers();

        if (delete_data)
        {
            if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\particle_properties.bin"))
                Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\particle_properties.bin");

            if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\neighbours_track.bin"))
                Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\neighbours_track.bin");

            if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\neighbours_list.bin"))
                Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\neighbours_list.bin");

            if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\densities.bin"))
                Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\densities.bin");

            FileStream saveFile = new FileStream(@"F:\UnityGames\SPHGPU\dataset\particle_properties.bin", FileMode.Create);
            saveFile.Close();
            FileStream saveFile1 = new FileStream(@"F:\UnityGames\SPHGPU\dataset\neighbours_track.bin", FileMode.Create);
            saveFile1.Close();
            FileStream saveFile2 = new FileStream(@"F:\UnityGames\SPHGPU\dataset\neighbours_list.bin", FileMode.Create);
            saveFile2.Close();
            FileStream saveFile3 = new FileStream(@"F:\UnityGames\SPHGPU\dataset\densities.bin", FileMode.Create);
            saveFile3.Close();
        }

        if (inference)
        {
            localAdd = IPAddress.Parse(connectionIP);
            listener = new TcpListener(IPAddress.Any, connectionPort);
            listener.Start();

            client = listener.AcceptTcpClient();
        }
        // print(Time.fixedDeltaTime);
    }

    // do not use fixed update
    private void FixedUpdate()
    {
        if (init_counter < init_interval)
        {
            init_counter++;
            use_python = false;
        }
        else if (init_counter == init_interval)
        {
            // only trigger once
            use_python = inference;
            if (use_python)
                print("Inference Started!");
            init_counter++;
        }

        float[] receivedData = new float[numberOfParticles];

        computeShaderSPH.Dispatch(clearHashGridKernel, dimensions * dimensions * dimensions / 100, 1, 1);
        computeShaderSPH.Dispatch(recalculateHashGridKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(buildNeighbourListKernel, numberOfParticles / 100, 1, 1);

        _densitiesBuffer.GetData(_densities);

        // float[] receiveData = SendAndReceive(_densities, _densities.Length);
        // print(receiveData.Length);

        /*
        FileStream saveFile = new FileStream(@"F:\UnityGames\SPHGPU\dataset\particle_properties.bin", FileMode.Append);
        var writer = new BinaryWriter(saveFile);
        FileStream saveFile1 = new FileStream(@"F:\UnityGames\SPHGPU\dataset\neighbours_track.bin", FileMode.Append);
        var writer1 = new BinaryWriter(saveFile1);
        FileStream saveFile2 = new FileStream(@"F:\UnityGames\SPHGPU\dataset\neighbours_list.bin", FileMode.Append);
        var writer2 = new BinaryWriter(saveFile2);
        FileStream saveFile3 = new FileStream(@"F:\UnityGames\SPHGPU\dataset\densities.bin", FileMode.Append);
        var writer3 = new BinaryWriter(saveFile3);

        if (save_data)
        {
            if (counter > save_interval)
            {
                // particle_properties
                _particlesBuffer.GetData(_particles);

                _velocitiesBuffer.GetData(_velocities); // vector3
                _forcesBuffer.GetData(_forces); // vector3
                _pressuresBuffer.GetData(_pressures); // float
                _densitiesBuffer.GetData(_densities); // float

                for (int i = 0; i < numberOfParticles; i++)
                {
                    writer.Write((float)_particles[i].position.x);
                    writer.Write((float)_particles[i].position.y);
                    writer.Write((float)_particles[i].position.z);
                    writer.Write((float)_velocities[i].x);
                    writer.Write((float)_velocities[i].y);
                    writer.Write((float)_velocities[i].z);
                    writer.Write((float)_forces[i].x);
                    writer.Write((float)_forces[i].y);
                    writer.Write((float)_forces[i].z);
                    writer.Write((float)_pressures[i]);
                    writer.Write((float)_densities[i]);
                }
                // neighbours_track
                _neighbourTrackerBuffer.GetData(_neighbourTracker); // int
                for (int i = 0; i < numberOfParticles; i++)
                    writer1.Write((Int16)_neighbourTracker[i]);

                // neighbours_list
                _neighbourListBuffer.GetData(_neighbourList); // int
                for (int i = 0; i < numberOfParticles; i++)
                {
                    for (int j=0; j < _neighbourTracker[i]; j++)
                    writer2.Write((Int16)_neighbourList[i*maximumParticlesPerCell*8 + j]);
                }
                
            }
        }
        */


        if (use_python)
        {
            // particle_properties
            _particlesBuffer.GetData(_particles); // particles -> position vector3
            _velocitiesBuffer.GetData(_velocities); // vector3
            _forcesBuffer.GetData(_forces); // vector3
            _pressuresBuffer.GetData(_pressures); // float
            _densitiesBuffer.GetData(_densities); // float

            // send data
            NetworkStream nwStream = client.GetStream();

            /*
            // send float
            float[] sendData = new float[numberOfParticles * 11];
            print("send float length: " + sendData.Length*4);

            for (int i = 0; i < numberOfParticles; i++)
            {
                sendData[i * 11 + 0] = (float)_particles[i].position.x;
                sendData[i * 11 + 1] = (float)_particles[i].position.y;
                sendData[i * 11 + 2] = (float)_particles[i].position.z;

                sendData[i * 11 + 3] = (float)_velocities[i].x;
                sendData[i * 11 + 4] = (float)_velocities[i].y;
                sendData[i * 11 + 5] = (float)_velocities[i].z;

                sendData[i * 11 + 6] = (float)_forces[i].x;
                sendData[i * 11 + 7] = (float)_forces[i].y;
                sendData[i * 11 + 8] = (float)_forces[i].z;

                sendData[i * 11 + 9] = (float)_pressures[i];
                sendData[i * 11 + 10] = (float)_densities[i];
            }

            byte[] byteArray = new byte[sendData.Length * 4];

            Buffer.BlockCopy(sendData, 0, byteArray, 0, byteArray.Length);

            print("send bytes length: " + byteArray.Length);

            nwStream.Write(byteArray, 0, byteArray.Length);
            */

            // send float--test
            float[] sendData = new float[11];
            byte[] byteArray = new byte[11 * 4];
            for (int i = 0; i < numberOfParticles; i++)
            {
                sendData[0] = (float)_particles[i].position.x;
                sendData[1] = (float)_particles[i].position.y;
                sendData[2] = (float)_particles[i].position.z;
                sendData[3] = (float)_velocities[i].x;
                sendData[4] = (float)_velocities[i].y;
                sendData[5] = (float)_velocities[i].z;
                sendData[6] = (float)_forces[i].x;
                sendData[7] = (float)_forces[i].y;
                sendData[8] = (float)_forces[i].z;
                sendData[9] = (float)_pressures[i];
                sendData[10] = (float)_densities[i];

                Buffer.BlockCopy(sendData, 0, byteArray, 0, byteArray.Length);

                nwStream.Write(byteArray, 0, byteArray.Length);
            }

            /*
            // send int 2
            _neighbourTrackerBuffer.GetData(_neighbourTracker); // int
            byteArray = new byte[_neighbourTracker.Length * 4];
            Buffer.BlockCopy(_neighbourTracker, 0, byteArray, 0, byteArray.Length);
            nwStream.Write(byteArray, 0, byteArray.Length);

            
            // send int 3
            int[] sendDataInt = new int[_neighbourTracker.Sum()];
            _neighbourListBuffer.GetData(_neighbourList); // int
            int k = 0;
            for (int i = 0; i < numberOfParticles; i++)
            {
                for (int j = 0; j < _neighbourList[i]; j++)
                {
                    sendDataInt[k] = _neighbourList[i * maximumParticlesPerCell * 8 + j];
                    k++;
                }
            }
            byteArray = new byte[sendDataInt.Length * 4];
            Buffer.BlockCopy(sendDataInt, 0, byteArray, 0, byteArray.Length);
            nwStream.Write(byteArray, 0, byteArray.Length);
            */
            // send int 3--test
            _neighbourListBuffer.GetData(_neighbourList); // int

            byteArray = new byte[_neighbourList.Length * 4];
            Buffer.BlockCopy(_neighbourList, 0, byteArray, 0, byteArray.Length);
            nwStream.Write(byteArray, 0, byteArray.Length);


            /*
            // receive data
            receivedData = new float[numberOfParticles];
            byte[] buffer = new byte[client.ReceiveBufferSize];
            int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);

            for (int i = 0; i < numberOfParticles; i++)
            {
                receivedData[i] = BitConverter.ToSingle(buffer, i*4);
            }
            // print(receivedData.Length);
            */

            // receive data--test
            receivedData = new float[2];
            byte[] buffer = new byte[client.ReceiveBufferSize];
            int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize);

            receivedData[0] = BitConverter.ToSingle(buffer, 0);
            receivedData[1] = BitConverter.ToSingle(buffer, 1);
        }


        computeShaderSPH.Dispatch(computeDensityPressureKernel, numberOfParticles / 100, 1, 1);


        if (use_python)
        {

            // set buffer
            // _densitiesBuffer.SetData(receivedData);
            // computeShaderSPH.SetBuffer(computeForcesKernel, "_densities", _densitiesBuffer);
            
        }

        computeShaderSPH.Dispatch(computeForcesKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(integrateKernel, numberOfParticles / 100, 1, 1);

        /*
        if (save_data)
        {
            if (counter > save_interval)
            {
                save_counter++;
                counter = 0;

                // densities
                _densitiesBuffer.GetData(_densities); // float
                for (int i = 0; i < numberOfParticles; i++)
                {
                    writer3.Write((float)_densities[i]);
                }
            }
            else
                counter++;
        }

        // close file
        saveFile.Flush();
        saveFile.Close();
        saveFile1.Flush();
        saveFile1.Close();
        saveFile2.Flush();
        saveFile2.Close();
        saveFile3.Flush();
        saveFile3.Close();
        */

        // material
        particleMaterial.SetFloat(SizeProperty, particleRenderSize);
        particleMaterial.SetBuffer(ParticlesBufferProperty, _particlesBuffer);

        Graphics.DrawMeshInstancedIndirect(particleMesh, 0, particleMaterial, new Bounds(Vector3.zero, new Vector3(100.0f, 100.0f, 100.0f)), _argsBuffer);

        print("num of frames ->");
    }

    private void RespawnParticles()
    {
        float initRandomOffset = 0.2f;

        _particles = new Particle[numberOfParticles];
        _densities = new float[numberOfParticles];
        _pressures = new float[numberOfParticles];
        _velocities = new Vector3[numberOfParticles];
        _forces = new Vector3[numberOfParticles];

        int particlesPerDimension = Mathf.CeilToInt(numberOfParticles/ particlePerRow/ particlePerRow);
        int count = 0;
        for (int x = 0; x < particlePerRow; x++)
            for (int y = 0; y < particlesPerDimension; y++)
                for (int z = 0; z < particlePerRow; z++)
                {
                    Vector3 startPos = new Vector3(dimensions - 1, dimensions - 1, dimensions - 1)
                        - new Vector3(x / initParticleDensity, y / initParticleDensity, z / initParticleDensity) - new Vector3(Random.Range(0, initRandomOffset), Random.Range(0f, initRandomOffset), Random.Range(0f, initRandomOffset));

                    _particles[count] = new Particle
                    {
                        position = startPos,
                        colorGradient = Color.white,
                    };
                    _densities[count] = 0.0f;
                    _pressures[count] = 0.0f;
                    _forces[count] = Vector3.zero;
                    _velocities[count] = Vector3.down * 50;
                    if (++count >= numberOfParticles)
                    {
                        return;
                    }
                }
    }

    private void FindKernels()
    {
        clearHashGridKernel = computeShaderSPH.FindKernel("ClearHashGrid");
        recalculateHashGridKernel = computeShaderSPH.FindKernel("RecalculateHashGrid");
        buildNeighbourListKernel = computeShaderSPH.FindKernel("BuildNeighbourList");
        computeDensityPressureKernel = computeShaderSPH.FindKernel("ComputeDensityPressure");
        computeForcesKernel = computeShaderSPH.FindKernel("ComputeForces");
        integrateKernel = computeShaderSPH.FindKernel("Integrate");
    }
    
    void InitComputeBuffers()
    {
        // set value
        computeShaderSPH.SetFloat("CellSize", radius * 2);
        computeShaderSPH.SetInt("Dimensions", dimensions);
        computeShaderSPH.SetInt("maximumParticlesPerCell", maximumParticlesPerCell);
        computeShaderSPH.SetFloat("radius", radius);
        computeShaderSPH.SetFloat("radius2", radius2);
        computeShaderSPH.SetFloat("radius3", radius3);
        computeShaderSPH.SetFloat("radius4", radius4);
        computeShaderSPH.SetFloat("radius5", radius4); // TODO: bug to fix
        computeShaderSPH.SetFloat("mass", mass);
        computeShaderSPH.SetFloat("mass2", mass2);
        computeShaderSPH.SetFloat("gasConstant", gasConstant);
        computeShaderSPH.SetFloat("restDensity", restDensity);
        computeShaderSPH.SetFloat("viscosityCoefficient", viscosityCoefficient);
        computeShaderSPH.SetFloat("damping", damping);
        computeShaderSPH.SetFloat("dt", dt);
        computeShaderSPH.SetFloats("gravity", gravity);
        computeShaderSPH.SetFloats("epsilon", Mathf.Epsilon);
        computeShaderSPH.SetFloat("pi", Mathf.PI);
        computeShaderSPH.SetFloat("stiffness", stiffness);

        // set buffers
        // argument
        uint[] args = {
            particleMesh.GetIndexCount(0), // get the index of particle index, uint
            (uint)numberOfParticles,
            0, // particleMesh.GetIndexStart(0),
            0, // particleMesh.GetBaseVertex(0),
            0
        };
        _argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        _argsBuffer.SetData(args);

        // particles
        _particlesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float) * 7);
        _particlesBuffer.SetData(_particles);
        computeShaderSPH.SetBuffer(recalculateHashGridKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(integrateKernel, "_particles", _particlesBuffer);

        // hash grid
        _hashGridTrackerBuffer = new ComputeBuffer(dimensions * dimensions * dimensions, sizeof(uint));
        computeShaderSPH.SetBuffer(clearHashGridKernel, "_hashGridTracker", _hashGridTrackerBuffer);
        computeShaderSPH.SetBuffer(recalculateHashGridKernel, "_hashGridTracker", _hashGridTrackerBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_hashGridTracker", _hashGridTrackerBuffer);
        
        _hashGrid = new uint[dimensions * dimensions * dimensions * maximumParticlesPerCell];
        _hashGridBuffer = new ComputeBuffer(dimensions * dimensions * dimensions * maximumParticlesPerCell, sizeof(uint));
        _hashGridBuffer.SetData(_hashGrid);
        computeShaderSPH.SetBuffer(recalculateHashGridKernel, "_hashGrid", _hashGridBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_hashGrid", _hashGridBuffer);

        // neighbours
        _neighbourTracker = new int[numberOfParticles];
        _neighbourTrackerBuffer = new ComputeBuffer(numberOfParticles, sizeof(int));
        _neighbourTrackerBuffer.SetData(_neighbourTracker);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_neighbourTracker", _neighbourTrackerBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_neighbourTracker", _neighbourTrackerBuffer);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_neighbourTracker", _neighbourTrackerBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_neighbourTracker", _neighbourTrackerBuffer);

        _neighbourList = new int[numberOfParticles * maximumParticlesPerCell * 8];
        _neighbourListBuffer = new ComputeBuffer(numberOfParticles * maximumParticlesPerCell * 8, sizeof(int));
        _neighbourListBuffer.SetData(_neighbourList);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_neighbourList", _neighbourListBuffer);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_neighbourList", _neighbourListBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_neighbourList", _neighbourListBuffer);

        // density
        _densitiesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float));
        _densitiesBuffer.SetData(_densities);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_densities", _densitiesBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_densities", _densitiesBuffer);

        // pressure
        _pressuresBuffer = new ComputeBuffer(numberOfParticles, sizeof(float));
        _pressuresBuffer.SetData(_pressures);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_pressures", _pressuresBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_pressures", _pressuresBuffer);

        // velocity
        _velocitiesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float) * 3);
        _velocitiesBuffer.SetData(_velocities);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_velocities", _velocitiesBuffer);
        computeShaderSPH.SetBuffer(integrateKernel, "_velocities", _velocitiesBuffer);

        // force
        _forcesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float) * 3);
        _forcesBuffer.SetData(_forces);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_forces", _forcesBuffer);
        computeShaderSPH.SetBuffer(integrateKernel, "_forces", _forcesBuffer);
    }

    private int Int16Sum(Int16[] data)
    {
        int sum = 0;
        foreach (var i in data)
            sum += i;
        return sum;
    }

    private float[] ServerRequest(float[] dataOut, int receiveLength)
    {
        float[] dataIn = SendAndReceive(dataOut, receiveLength);
        return dataIn;
    }

    private float[] SendAndReceive(float[] dataOut, int receiveLength)
    {
        //initialize socket
        float[] floatsReceived;
        socketClient = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        socketClient.Connect(ip, port);

        var byteArray = new byte[dataOut.Length * 4];
        Buffer.BlockCopy(dataOut, 0, byteArray, 0, byteArray.Length);
        socketClient.Send(byteArray);

        //allocate and receive bytes
        byte[] bytes = new byte[receiveLength * 4];
        int idxUsedBytes = socketClient.Receive(bytes);

        //convert bytes to floats
        floatsReceived = new float[idxUsedBytes / 4];
        Buffer.BlockCopy(bytes, 0, floatsReceived, 0, idxUsedBytes);

        client.Close();
        return floatsReceived;
    }

    private void OnDestroy()
    {
        // release buffers
        _particlesBuffer.Dispose();
        _argsBuffer.Dispose();
        _neighbourListBuffer.Dispose();
        _neighbourTrackerBuffer.Dispose();
        _hashGridBuffer.Dispose();
        _hashGridTrackerBuffer.Dispose();
        _densitiesBuffer.Dispose();
        _pressuresBuffer.Dispose();
        _velocitiesBuffer.Dispose();
        _forcesBuffer.Dispose();

        if (save_data)
            print("Saved " + save_counter + " frames.");

        if (inference)
            listener.Stop();
    }

}