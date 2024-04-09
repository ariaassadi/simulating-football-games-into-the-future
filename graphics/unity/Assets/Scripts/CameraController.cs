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

    void Start()
    {
        moveSpeed = 10f;
        rotationSpeed = 100f;
        verticalSpeed = 10f;
        mainCamera = GetComponent<Camera>();

        // Set the initial camera position and rotation
        lastKeyPressed = KeyCode.N;
        SideView();
    }

    void Update()
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

    void MoveCamera()
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

    void BirdsEyeView()
    {
        // Move the camera to a birds eye view
        transform.position = new Vector3(57.5f, 100f, 34f);
        transform.rotation = Quaternion.Euler(90, 0, 0);

        mainCamera.orthographic = true;
        mainCamera.orthographicSize = 34f;
    }

    void SideView()
    {
        mainCamera.orthographic = false;
        mainCamera.fieldOfView = 80f;
        transform.position = new Vector3(52.2f, 20f, 78f);
        transform.rotation = Quaternion.Euler(35, 180, 0);
    }

    // Add options to change moveSpeed, rotationSpeed, and verticalSpeed

    public void IncrementMoveSpeed()
    {
        moveSpeed += 1f;
    }

    public void DecrementMoveSpeed()
    {
        moveSpeed -= 1f;
    }

    public void IncrementRotationSpeed()
    {
        rotationSpeed += 10f;
    }

    public void DecrementRotationSpeed()
    {
        rotationSpeed -= 10f;
    }

    public void IncrementVerticalSpeed()
    {
        verticalSpeed += 1f;
    }

    public void DecrementVerticalSpeed()
    {
        verticalSpeed -= 1f;
    }

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

    public void SetSetting(string setting, float value)
    {
        switch (setting)
        {
            case "MoveSpeed":
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

    public void ResetSpeeds()
    {
        moveSpeed = 10f;
        rotationSpeed = 100f;
        verticalSpeed = 10f;
    }

    public float GetCameraSpeed()
    {
        return moveSpeed;
    }

}
