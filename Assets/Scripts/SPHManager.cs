using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;




public class SPHManager : MonoBehaviour
{
    public struct Particle
    {
        public Vector3 position;

        public Vector4 colorGradient;
    }

    [SerializeField]
    [Header("Particle properties")]
    public float radius = 2f;
    public Mesh particleMesh;
    Transform pointPrefab;
    public float particleRenderSize = 1f;
    public Material material;
    public float mass = 1f;
    public float gasConstant = 2000f;
    public float restDensity = 2f;
    public float viscosityCoefficient = 1.2f;
    public float[] gravity = { 0.0f, -9.81f, 0.0f };
    public float damping = -0.5f;
    public float dt = 0.01f;

    private float volume;
    private float radius2;
    private float radius3;
    private float radius4;
    private float radius5;
    private float mass2;
    public float stiffness = 1f;
    private int currParticleNum { get { return _particles.Length; } }



    [Header("Simulation space properties")]
    public int numberOfParticles = 2000;
    public int dimensions = 100;
    public int maximumParticlesPerCell = 100;

    [Header("Debug information")]
    [Tooltip("Tracks how many neighbours each particleIndex has in" + nameof(_neighbourList))]

    public int[] _neighbourTracker;
    private Particle[] _particles;

    private int[] _neighbourList;
    private uint[] _hashGrid;
    private float[] _densities;
    private float[] _pressures;
    private Vector3[] _velocities;
    private Vector3[] _forces;

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
    private static readonly int SizeProperty = Shader.PropertyToID("_size");
    private static readonly int ParticlesBufferProperty = Shader.PropertyToID("_particlesBuffer");

    public ComputeShader computeShaderSPH;
    private int clearHashGridKernel;
    private int recalculateHashGridKernel;
    private int buildNeighbourListKernel;
    private int computeDensityPressureKernel;
    private int computeForcesKernel;
    private int integrateKernel;


    private void Awake()
    {
        Application.targetFrameRate = 60;
        radius2 = radius * radius;
        radius3 = radius * radius2;
        radius4 = radius2 * radius2;
        radius5 = radius4 * radius;
        mass2 = mass * mass;

        RespawnParticles();
        FindKernels();
        InitComputeShader();
        InitComputeBuffers();

    }

    #region Initilisation
    private void RespawnParticles()
    {

        _particles = new Particle[numberOfParticles];
        _densities = new float[numberOfParticles];
        _pressures = new float[numberOfParticles];
        _velocities = new Vector3[numberOfParticles];
        _forces = new Vector3[numberOfParticles];

        int particlesPerDimension = Mathf.CeilToInt(Mathf.Pow(numberOfParticles, 1f / 3f));
        volume = Mathf.Pow(2 * radius, 3);
        int counter = 0;
        while (counter < numberOfParticles / 2)
        {
            for (int x = 0; x < particlesPerDimension; x++)
                for (int y = 0; y < particlesPerDimension; y++)
                    for (int z = 0; z < particlesPerDimension; z++)
                    {
                        Vector3 startPos = new Vector3(dimensions - 1, dimensions - 1, dimensions - 1)
                            - new Vector3(x / 2f, y / 2f, z / 2f) - new Vector3(Random.Range(0, 0.01f), Random.Range(0f, 0.01f), Random.Range(0f, 0.01f));

                        _particles[counter] = new Particle
                        {
                            position = startPos,
                            colorGradient = Color.white,
                        };
                        _densities[counter] = -1f;
                        _pressures[counter] = 0.0f;
                        _forces[counter] = Vector3.zero;
                        _velocities[counter] = Vector3.down * 50;
                        if (++counter == numberOfParticles)
                        {
                            return;
                        }
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
    private void InitComputeShader()
    {
        computeShaderSPH.SetFloat("CellSize", radius * 2);
        computeShaderSPH.SetInt("Dimensions", dimensions);
        computeShaderSPH.SetInt("maximumParticlesPerCell", maximumParticlesPerCell);
        computeShaderSPH.SetFloat("radius", radius);
        computeShaderSPH.SetFloat("radius2", radius2);
        computeShaderSPH.SetFloat("radius3", radius3);
        computeShaderSPH.SetFloat("radius4", radius4);
        computeShaderSPH.SetFloat("radius5", radius4);
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


    }

    void InitComputeBuffers()
    {
        uint[] args = {
        particleMesh.GetIndexCount(0),
        (uint)numberOfParticles,
        particleMesh.GetIndexStart(0),
        particleMesh.GetBaseVertex(0),
        0
        };
        _argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        _argsBuffer.SetData(args);
        _particlesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float) * (3 + 4));
        _particlesBuffer.SetData(_particles);


        _neighbourList = new int[numberOfParticles * maximumParticlesPerCell * 8];
        _neighbourTracker = new int[numberOfParticles];

        _hashGrid = new uint[dimensions * dimensions * dimensions * maximumParticlesPerCell];

        _neighbourListBuffer = new ComputeBuffer(numberOfParticles * maximumParticlesPerCell * 8, sizeof(int));
        _neighbourListBuffer.SetData(_neighbourList);
        _neighbourTrackerBuffer = new ComputeBuffer(numberOfParticles, sizeof(int));
        _neighbourTrackerBuffer.SetData(_neighbourTracker);

        _hashGridBuffer = new ComputeBuffer(dimensions * dimensions * dimensions * maximumParticlesPerCell, sizeof(uint));
        _hashGridBuffer.SetData(_hashGrid);
        _hashGridTrackerBuffer = new ComputeBuffer(dimensions * dimensions * dimensions, sizeof(uint));

        _densitiesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float));
        _densitiesBuffer.SetData(_densities);
        _pressuresBuffer = new ComputeBuffer(numberOfParticles, sizeof(float));
        _pressuresBuffer.SetData(_pressures);

        _velocitiesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float) * 3);
        _velocitiesBuffer.SetData(_velocities);
        _forcesBuffer = new ComputeBuffer(numberOfParticles, sizeof(float) * 3);
        _forcesBuffer.SetData(_forces);

        computeShaderSPH.SetBuffer(clearHashGridKernel, "_hashGridTracker", _hashGridTrackerBuffer);


        computeShaderSPH.SetBuffer(recalculateHashGridKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(recalculateHashGridKernel, "_hashGrid", _hashGridBuffer);
        computeShaderSPH.SetBuffer(recalculateHashGridKernel, "_hashGridTracker", _hashGridTrackerBuffer);

        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_hashGrid", _hashGridBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_hashGridTracker", _hashGridTrackerBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_neighbourList", _neighbourListBuffer);
        computeShaderSPH.SetBuffer(buildNeighbourListKernel, "_neighbourTracker", _neighbourTrackerBuffer);


        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_neighbourTracker", _neighbourTrackerBuffer);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_neighbourList", _neighbourListBuffer);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_densities", _densitiesBuffer);
        computeShaderSPH.SetBuffer(computeDensityPressureKernel, "_pressures", _pressuresBuffer);


        computeShaderSPH.SetBuffer(computeForcesKernel, "_neighbourTracker", _neighbourTrackerBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_neighbourList", _neighbourListBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_densities", _densitiesBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_pressures", _pressuresBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_velocities", _velocitiesBuffer);
        computeShaderSPH.SetBuffer(computeForcesKernel, "_forces", _forcesBuffer);

        computeShaderSPH.SetBuffer(integrateKernel, "_particles", _particlesBuffer);
        computeShaderSPH.SetBuffer(integrateKernel, "_forces", _forcesBuffer);
        computeShaderSPH.SetBuffer(integrateKernel, "_velocities", _velocitiesBuffer);


    }

    #endregion
    // do not use fixed update
    private void Update()
    {
        computeShaderSPH.Dispatch(clearHashGridKernel, dimensions * dimensions * dimensions / 100, 1, 1);
        computeShaderSPH.Dispatch(recalculateHashGridKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(buildNeighbourListKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(computeDensityPressureKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(computeForcesKernel, numberOfParticles / 100, 1, 1);
        computeShaderSPH.Dispatch(integrateKernel, numberOfParticles / 100, 1, 1);
        material.SetFloat(SizeProperty, particleRenderSize);
        material.SetBuffer(ParticlesBufferProperty, _particlesBuffer);
        Graphics.DrawMeshInstancedIndirect(particleMesh, 0, material, new Bounds(Vector3.zero, new Vector3(100.0f, 100.0f, 100.0f)), _argsBuffer);
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
}