using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
}
