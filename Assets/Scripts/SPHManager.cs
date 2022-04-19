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

    private int save_interval = 100;
    private int counter = 0;

    private void Awake()
    {
        Application.targetFrameRate = 60;

        radius2 = Mathf.Pow(radius, 2);
        radius3 = Mathf.Pow(radius, 3);
        radius4 = Mathf.Pow(radius, 4);
        radius5 = Mathf.Pow(radius, 5);
        mass2 = Mathf.Pow(mass, 2);

        RespawnParticles();
        FindKernels();
        InitComputeBuffers();

        if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\input.bin"))
        {
            Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\input.bin");
        }

        FileStream saveFile = new FileStream(@"F:\UnityGames\SPHGPU\dataset\input.bin", FileMode.Create);
        saveFile.Close();
    }

    // do not use fixed update
    private void Update()
    {
        Vector3[] pre_position = new Vector3[numberOfParticles];
        FileStream saveFile = new FileStream(@"F:\UnityGames\SPHGPU\dataset\input.bin", FileMode.Append);
        var writer = new BinaryWriter(saveFile);

        computeShaderSPH.Dispatch(clearHashGridKernel, dimensions * dimensions * dimensions / 100, 1, 1);
        computeShaderSPH.Dispatch(recalculateHashGridKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(buildNeighbourListKernel, numberOfParticles / 100, 1, 1);

        if (counter > save_interval)
        {
            // get data
            _particlesBuffer.GetData(_particles);
            for (int i = 0; i < numberOfParticles; i++)
            {
                pre_position[i] = _particles[i].position; // vector3
            }
            _velocitiesBuffer.GetData(_velocities); // vector3
            _forcesBuffer.GetData(_forces); // vector3
            _pressuresBuffer.GetData(_pressures); // float
            _densitiesBuffer.GetData(_densities); // float

            // save data
            for (int i = 0; i < numberOfParticles; i++)
            {
                writer.Write((float)pre_position[i][0]);
                writer.Write((float)pre_position[i][1]);
                writer.Write((float)pre_position[i][2]);
                writer.Write((float)_velocities[i][0]);
                writer.Write((float)_velocities[i][1]);
                writer.Write((float)_velocities[i][2]);
                writer.Write((float)_forces[i][0]);
                writer.Write((float)_forces[i][1]);
                writer.Write((float)_forces[i][2]);
                writer.Write((float)_pressures[i]);
                writer.Write((float)_densities[i]);
            }
        }



        computeShaderSPH.Dispatch(computeDensityPressureKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(computeForcesKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(integrateKernel, numberOfParticles / 100, 1, 1);

        if (counter > save_interval)
        {
            // get data
            _densitiesBuffer.GetData(_densities); // float

            // save data
            for (int i = 0; i < numberOfParticles; i++)
            {
                writer.Write((float)_densities[i]);
            }
        }

        // close file
        saveFile.Flush();
        saveFile.Close();
        counter++;


        // material
        particleMaterial.SetFloat(SizeProperty, particleRenderSize);
        particleMaterial.SetBuffer(ParticlesBufferProperty, _particlesBuffer);

        Graphics.DrawMeshInstancedIndirect(particleMesh, 0, particleMaterial, new Bounds(Vector3.zero, new Vector3(100.0f, 100.0f, 100.0f)), _argsBuffer);
    }

    private void RespawnParticles()
    {
        float initParticleDensity = 1.5f;
        float initRandomOffset = 0.2f;
        int particlePerRow = 50;

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



    }

    private void Save2Bin()
    {
        if (Directory.Exists(@"F:\UnityGames\SPHGPU\dataset\input.bin"))
        {
            Directory.Delete(@"F:\UnityGames\SPHGPU\dataset\input.bin");
        }
        using FileStream saveFile = File.Create(@"F:\UnityGames\SPHGPU\dataset\input.bin");

        /*
        foreach (float f in input_data)
        {
            saveFile.Write(BitConverter.GetBytes(f), 0, sizeof(float));
        }
        */

        saveFile.Flush();
        saveFile.Close();
    }

}