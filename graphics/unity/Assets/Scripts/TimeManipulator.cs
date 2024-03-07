using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TimeManager : MonoBehaviour
{
    [SerializeField] private GameObject playerSpawner;

    private float timeManipulationSpeed = 0.04f;

    private float timer = 0f;
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.K))
        {
            playerSpawner.GetComponent<CreateAndMovePlayers>().PlayPause();
        }
        timer += Time.deltaTime;

        if (timer >= timeManipulationSpeed)
        {
            timer = 0f;
            if (Input.GetKey(KeyCode.J))
            {
                playerSpawner.GetComponent<CreateAndMovePlayers>().FastBackward();
            }
            else if (Input.GetKey(KeyCode.Period))
            {
                playerSpawner.GetComponent<CreateAndMovePlayers>().StepForward();
            }
            else if (Input.GetKey(KeyCode.L))
            {
                playerSpawner.GetComponent<CreateAndMovePlayers>().FastForward();
            }
            else if (Input.GetKey(KeyCode.Comma))
            {
                playerSpawner.GetComponent<CreateAndMovePlayers>().StepBackward();
            }
        }


    }
}
