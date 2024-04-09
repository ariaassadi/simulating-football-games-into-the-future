using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class CameraController : MonoBehaviour
{
    private float moveSpeed;
    private float rotationSpeed;
    private float verticalSpeed;
    private Camera mainCamera;

    KeyCode lastKeyPressed;

    private void Start()
    {
        moveSpeed = 10f;
        rotationSpeed = 100f;
        verticalSpeed = 10f;
        mainCamera = GetComponent<Camera>();

        // Set the initial camera position and rotation
        lastKeyPressed = KeyCode.N;
        SideView();
    }

    private void Update()
    {

        if (Input.GetKeyDown(KeyCode.B))
        {
            lastKeyPressed = KeyCode.B;
            BirdsEyeView();
        }
        else if (Input.GetKeyDown(KeyCode.N))
        {
            lastKeyPressed = KeyCode.N;
            SideView();
        }

        // Camera movement only if the camera is not in birds eye view
        if (lastKeyPressed != KeyCode.B)
            MoveCamera();
    }

    private void MoveCamera()
    {

        bool rotateButtonPressed = Input.GetKey(KeyCode.Mouse1);

        // Rotation based on mouse movement
        if (rotateButtonPressed)
        {
            // Hide and lock the cursor when rotating the camera
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
            float mouseDown = Input.GetAxis("Mouse X") * rotationSpeed * Time.deltaTime;
            float mouseUp = Input.GetAxis("Mouse Y") * rotationSpeed * Time.deltaTime;

            Vector3 current_rotation = transform.localEulerAngles;
            Vector3 mouse_rotation;

            // Prevent the camera from rotating more than 90 degrees up or down
            if (Math.Abs(current_rotation.x - mouseUp) > 90f && Math.Abs(current_rotation.x - mouseUp) < 270f)
            {
                mouse_rotation = new Vector3(0, mouseDown, 0);
            }
            else
            {
                mouse_rotation = new Vector3(-mouseUp, mouseDown, 0);
            }

            transform.rotation = Quaternion.Euler(current_rotation + mouse_rotation);
        }
        else
        {
            // Show and unlock the cursor when not rotating the camera
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        // Forward and backward movement based on keyboard input
        float verticalInput = Input.GetAxis("Vertical");
        Vector3 forwardDirection = transform.forward;
        forwardDirection.y = 0; // Ensure movement is only along the XZ plane
        forwardDirection.Normalize(); // Normalize to maintain consistent speed regardless of camera rotation
        transform.position += forwardDirection * verticalInput * moveSpeed * Time.deltaTime;

        // Left and right movement based on keyboard input
        float horizontalInput = Input.GetAxis("Horizontal");
        Vector3 rightDirection = transform.right;
        rightDirection.y = 0; // Ensure movement is only along the XZ plane
        rightDirection.Normalize(); // Normalize to maintain consistent speed regardless of camera rotation
        transform.position += rightDirection * horizontalInput * moveSpeed * Time.deltaTime;

        // Up and down movement based on keyboard input
        Vector3 verticalMovement = Vector3.zero;
        if (Input.GetKey(KeyCode.Space)) // Move up
        {
            verticalMovement = new Vector3(0, 1f, 0);
        }
        if (Input.GetKey(KeyCode.LeftShift)) // Move down
        {
            verticalMovement = new Vector3(0, -1f, 0);
        }
        transform.position += verticalMovement * verticalSpeed * Time.deltaTime;
    }

    /// <summary>
    /// Move the camera to a birds eye view, which is orthographic
    /// </summary>
    private void BirdsEyeView()
    {
        // Move the camera to a birds eye view
        transform.position = new Vector3(57.5f, 100f, 34f);
        transform.rotation = Quaternion.Euler(90, 0, 0);

        mainCamera.orthographic = true;
        mainCamera.orthographicSize = 34f;
    }

    /// <summary>
    /// Move the camera to a side view, which is the neutral/initial position
    /// </summary>
    private void SideView()
    {
        mainCamera.orthographic = false;
        mainCamera.fieldOfView = 80f;
        transform.position = new Vector3(52.2f, 20f, 78f);
        transform.rotation = Quaternion.Euler(35, 180, 0);
    }

    // Add options to change moveSpeed, rotationSpeed, and verticalSpeed

    /// <summary>
    /// Increase the specified setting by 1
    /// </summary>
    /// <param name="setting">
    /// The setting to increase
    /// </param>
    public void IncreaseSetting(string setting)
    {
        switch (setting)
        {
            case "HorizontalSpeed":
                moveSpeed += 1f;
                break;
            case "RotationSpeed":
                rotationSpeed += 1f;
                break;
            case "VerticalSpeed":
                verticalSpeed += 1f;
                break;
        }
    }

    /// <summary>
    /// Decrease the specified setting by 1
    /// </summary>
    /// <param name="setting">
    /// The setting to decrease
    /// </param>
    public void DecreaseSetting(string setting)
    {
        switch (setting)
        {
            case "HorizontalSpeed":
                moveSpeed -= 1f;
                break;
            case "RotationSpeed":
                rotationSpeed -= 1f;
                break;
            case "VerticalSpeed":
                verticalSpeed -= 1f;
                break;
        }
    }

    /// <summary>
    /// Get the value of the specified setting
    /// </summary>
    /// <param name="setting">
    /// The setting to retrieve
    /// </param>
    /// <returns>
    /// The value of the specified setting
    /// </returns>
    public float GetSetting(string setting)
    {
        switch (setting)
        {
            case "HorizontalSpeed":
                return moveSpeed;
            case "RotationSpeed":
                return rotationSpeed;
            case "VerticalSpeed":
                return verticalSpeed;
            default:
                return 0f;
        }
    }

    /// <summary>
    /// Set the value of the specified setting
    /// </summary>
    /// <param name="setting">
    /// The setting to change
    /// </param>
    /// <param name="value">
    /// The new value for the setting
    /// </param>
    public void SetSetting(string setting, float value)
    {
        switch (setting)
        {
            case "HorizontalSpeed":
                moveSpeed = value;
                break;
            case "RotationSpeed":
                rotationSpeed = value;
                break;
            case "VerticalSpeed":
                verticalSpeed = value;
                break;
        }
    }

    /// <summary>
    /// Reset the specified setting to its default value
    /// </summary>
    /// <param name="setting">
    /// The setting to reset
    /// </param>
    public void ResetSetting(string setting)
    {
        switch (setting)
        {
            case "MorizontalSpeed":
                moveSpeed = 10f;
                break;
            case "RotationSpeed":
                rotationSpeed = 100f;
                break;
            case "VerticalSpeed":
                verticalSpeed = 10f;
                break;
        }
    }
}
