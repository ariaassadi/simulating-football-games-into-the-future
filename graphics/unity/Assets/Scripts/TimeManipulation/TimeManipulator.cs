using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;

using GameVisualization;

public class TimeManager : MonoBehaviour
{
    [SerializeField] private GameObject gameManager;

    private float timeManipulationSpeed = 0.04f;

    private float timer = 0f;
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.K))
        {
            gameManager.GetComponent<GameManager>().PlayPause();
        }
        timer += Time.deltaTime;

        if (timer >= timeManipulationSpeed)
        {
            timer = 0f;
            if (Input.GetKey(KeyCode.J))
            {
                gameManager.GetComponent<GameManager>().FastBackward();
            }
            else if (Input.GetKey(KeyCode.Period))
            {
                gameManager.GetComponent<GameManager>().StepForward();
            }
            else if (Input.GetKey(KeyCode.L))
            {
                gameManager.GetComponent<GameManager>().FastForward();
            }
            else if (Input.GetKey(KeyCode.Comma))
            {
                gameManager.GetComponent<GameManager>().StepBackward();
            }
        }
    }

    private void ManipulateTime(Action manipulationAction)
    {
        timer += Time.deltaTime;

        if (timer >= timeManipulationSpeed)
        {
            timer = 0f;
            manipulationAction.Invoke();
        }
    }

    public void FastBackward()
    {
        // timer += Time.deltaTime;

        // if (timer >= timeManipulationSpeed)
        // {
        //     timer = 0f;
        while (Input.GetMouseButton(0))
        {
            gameManager.GetComponent<GameManager>().FastBackward();
        }

        Debug.Log("Button is not held");
        // }
    }
    // public void FastBackward()
    // {
    //     ManipulateTime(() => gameManager.GetComponent<GameManager>().FastBackward());
    // }

    public void FastForward()
    {
        // if pointer is down
        if (Input.GetKey(KeyCode.Mouse0))
        {
            ManipulateTime(() => gameManager.GetComponent<GameManager>().FastForward());
        }
    }

    public void StepBackward()
    {
        ManipulateTime(() => gameManager.GetComponent<GameManager>().StepBackward());
    }

    public void StepForward()
    {
        ManipulateTime(() => gameManager.GetComponent<GameManager>().StepForward());
    }

    public void PlayPause()
    {
        gameManager.GetComponent<GameManager>().PlayPause();
    }

}
