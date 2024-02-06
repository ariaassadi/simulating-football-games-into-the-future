using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnSphere : MonoBehaviour
{
    public GameObject playerObject;

    void Start()
    {
        Instantiate(playerObject, new Vector3(10.0f, 0.0f, 10.0f), transform.rotation);
    }

}

