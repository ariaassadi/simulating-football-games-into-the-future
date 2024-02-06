using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

public class MoveObject : MonoBehaviour
{
    private string filePath = "Assets/Static/Coordinates/coordinates.txt";
    public float moveSpeed = 10.0f;

    private int i = 0;

    private string[] lines;

    private Vector3 prevPosition;

    private float timer = 0f;

    private float updateInterval = 1.0f;

    void Start()
    {
        lines = File.ReadAllLines(filePath);
        prevPosition = transform.position;
    }
    void Update()
    {
        // Update the timer
        timer += Time.deltaTime;

        // Check if the specified interval has passed
        if (timer >= updateInterval)
        {
            // Reset the timer
            timer = 0f;

            // Update the position based on the current element in the movements array
            MoveObjectWithCoordinates(lines[i]);
            // Increment the index
            i = (i + 1) % lines.Length;
        }
    }

    void MoveObjectWithCoordinates(string line)
    {
        string[] coordinates = line.Split(',');
        if (coordinates.Length == 3)
        {
            float x = float.Parse(coordinates[0]);
            float y = float.Parse(coordinates[1]);
            float z = float.Parse(coordinates[2]);

            Vector3 targetPosition = new Vector3(x, y, z);

            // transform.position = Vector3.MoveTowards(transform.position, targetPosition, Vector3.Distance(prevPosition, targetPosition));
            StartCoroutine(MoveObjectCoroutine(targetPosition));
            // StartCoroutine(MoveObjectCoroutine(targetPosition));
            // prevPosition = targetPosition;
        }
    }

    IEnumerator MoveObjectCoroutine(Vector3 targetPosition)
    {
        Vector3 initialPosition = transform.position;
        moveSpeed = Vector3.Distance(initialPosition, targetPosition);

        float elapsedTime = 0f;
        while (elapsedTime < 1.0f)
        {
            // Interpolate the position over time using Lerp
            transform.position = Vector3.Lerp(initialPosition, targetPosition, elapsedTime);

            // Increment the timer
            elapsedTime += Time.deltaTime;

            yield return null;
        }

        // Ensure the object reaches the exact target position
        transform.position = targetPosition;
    }
}
