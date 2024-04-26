using System;
using UnityEngine;

using Utils;

public class CameraController : MonoBehaviour
{
    // Reference to the settings used by the camera
    private Settings settings;

    // Temporary boost multiplier for camera movement
    private float tmpBoost = 1;

    // Reference to the main camera
    private Camera mainCamera;

    // Path to the settings file
    private string pathToSettings;

    // Last key pressed by the user
    KeyCode lastKeyPressed;

    private void Start()
    {
        pathToSettings = Application.persistentDataPath + "/settings.json";
        // Load settings from file or use default settings
        LoadSettings();

        // Get the main camera component
        mainCamera = GetComponent<Camera>();

        // Set the initial camera position and rotation
        lastKeyPressed = KeyCode.N;
        SideView();
    }

    private void Update()
    {
        // Check for key presses to switch camera views
        HandleViewSwitching();

        // Move the camera if it's not in birds eye view
        if (lastKeyPressed != KeyCode.B)
            MoveCamera();
    }

    // Handles switching between different camera views
    private void HandleViewSwitching()
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
    }

    // Moves the camera based on user input
    private void MoveCamera()
    {
        // Check if the rotate button is pressed
        bool rotateButtonPressed = Input.GetKey(KeyCode.Mouse1);

        // Handle camera rotation
        if (rotateButtonPressed)
        {
            // Rotate the camera based on mouse movement
            RotateCamera();
        }
        else
        {
            // Show and unlock the cursor when not rotating the camera
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        // Apply temporary movement boost
        ApplyTemporaryBoost();

        // Move the camera based on keyboard input
        MoveWithKeyboard();
    }

    // Rotates the camera based on mouse movement
    private void RotateCamera()
    {
        // Hide and lock the cursor when rotating the camera
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        // Calculate rotation based on mouse movement
        float mouseDown = Input.GetAxis("Mouse X") * settings.rotationSpeed * Time.deltaTime;
        float mouseUp = Input.GetAxis("Mouse Y") * settings.rotationSpeed * Time.deltaTime;

        // Apply rotation to the camera
        Vector3 current_rotation = transform.localEulerAngles;
        Vector3 mouse_rotation;

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

    // Applies temporary boost to camera movement
    private void ApplyTemporaryBoost()
    {
        tmpBoost = Input.GetKey(KeyCode.LeftControl) ? 4 : 1;
    }

    // Moves the camera based on keyboard input
    private void MoveWithKeyboard()
    {
        // Forward and backward movement
        float verticalInput = Input.GetAxis("Vertical");
        Vector3 forwardDirection = transform.forward;
        forwardDirection.y = 0; // Ensure movement is only along the XZ plane
        forwardDirection.Normalize(); // Normalize to maintain consistent speed regardless of camera rotation
        transform.position += forwardDirection * verticalInput * settings.horizontalSpeed * tmpBoost * Time.deltaTime;

        // Left and right movement
        float horizontalInput = Input.GetAxis("Horizontal");
        Vector3 rightDirection = transform.right;
        rightDirection.y = 0; // Ensure movement is only along the XZ plane
        rightDirection.Normalize(); // Normalize to maintain consistent speed regardless of camera rotation
        transform.position += rightDirection * horizontalInput * settings.horizontalSpeed * tmpBoost * Time.deltaTime;

        // Up and down movement
        Vector3 verticalMovement = Vector3.zero;
        if (Input.GetKey(KeyCode.Space)) // Move up
        {
            verticalMovement = new Vector3(0, 1f, 0);
        }
        if (Input.GetKey(KeyCode.LeftShift)) // Move down
        {
            verticalMovement = new Vector3(0, -1f, 0);
        }
        transform.position += verticalMovement * settings.verticalSpeed * tmpBoost / 2 * Time.deltaTime;
    }

    // Move the camera to a birds eye view
    private void BirdsEyeView()
    {
        // Set camera position and rotation for birds eye view
        transform.position = new Vector3(57.5f, 100f, 34f);
        transform.rotation = Quaternion.Euler(90, 0, 0);

        // Enable orthographic mode and set orthographic size
        mainCamera.orthographic = true;
        mainCamera.orthographicSize = 34f;
    }

    // Move the camera to a side view
    private void SideView()
    {
        // Disable orthographic mode and set field of view
        mainCamera.orthographic = false;
        mainCamera.fieldOfView = 80f;

        // Set camera position and rotation for side view
        transform.position = new Vector3(52.2f, 20f, 78f);
        transform.rotation = Quaternion.Euler(35, 180, 0);
    }

    // Loads settings from file or uses default settings
    public void LoadSettings()
    {
        if (settings == null)
        {
            if (System.IO.File.Exists(pathToSettings))
            {
                Debug.Log("Settings file found at: " + pathToSettings);
                string json = System.IO.File.ReadAllText(pathToSettings);
                settings = JsonParser.GetSettingsFromJson(json);
            }
            else
            {
                Debug.Log("Settings file not found. Using default settings.");
                string json = Resources.Load<TextAsset>("DefaultSettings").text;
                settings = JsonParser.GetSettingsFromJson(json);
            }
        }
        else
            Debug.Log("Settings already loaded");

    }

    // Saves current settings to file
    public void SaveSettings()
    {
        string json = JsonParser.GetJsonFromSettings(settings);
        JsonParser.WriteJsonToFile(json, pathToSettings);
    }

    // Increases the specified setting by 1 
    public void IncreaseSetting(string setting)
    {
        while (settings == null)
        {
            Debug.LogWarning("Settings is null");
        }
        switch (setting)
        {
            case "HorizontalSpeed":
                settings.horizontalSpeed += 1f;
                break;
            case "RotationSpeed":
                settings.rotationSpeed += 10f;
                break;
            case "VerticalSpeed":
                settings.verticalSpeed += 1f;
                break;
        }
    }

    // Decreases the specified setting by 1
    public void DecreaseSetting(string setting)
    {
        while (settings == null)
        {
            Debug.LogWarning("Settings is null");
        }
        switch (setting)
        {
            case "HorizontalSpeed":
                settings.horizontalSpeed -= 1f;
                break;
            case "RotationSpeed":
                settings.rotationSpeed -= 1f;
                break;
            case "VerticalSpeed":
                settings.verticalSpeed -= 1f;
                break;
        }
    }

    // Gets the value of the specified setting
    public float GetSetting(string setting)
    {
        switch (setting)
        {
            case "HorizontalSpeed":
                return settings.horizontalSpeed;
            case "RotationSpeed":
                return settings.rotationSpeed;
            case "VerticalSpeed":
                return settings.verticalSpeed;
            default:
                return 0f;
        }
    }

    // Sets the value of the specified setting
    public void SetSetting(string setting, float value)
    {
        while (settings == null)
        {
            Debug.LogWarning("Settings is null");
        }
        switch (setting)
        {
            case "HorizontalSpeed":
                settings.horizontalSpeed = value;
                break;
            case "RotationSpeed":
                settings.rotationSpeed = value;
                break;
            case "VerticalSpeed":
                settings.verticalSpeed = value;
                break;
        }
    }

    // Resets the specified setting to its default value
    public void ResetSetting(string setting)
    {
        while (settings == null)
        {
            Debug.LogWarning("Settings is null");
            //wait for 10 ms
            // if waited for 5 seconds, debug log error and return
        }
        switch (setting)
        {
            case "HorizontalSpeed":
                settings.horizontalSpeed = 10f;
                break;
            case "RotationSpeed":
                settings.rotationSpeed = 100f;
                break;
            case "VerticalSpeed":
                settings.verticalSpeed = 10f;
                break;
        }
    }

}
