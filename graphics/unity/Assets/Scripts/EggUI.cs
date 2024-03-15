using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class EggUI : MonoBehaviour
{
    private Camera mainCamera;
    [SerializeField] private TMP_Text textComponent;

    private string jerseyNumber;
    private string playerName;

    private void Start()
    {
        mainCamera = Camera.main;
        if (mainCamera == null || textComponent == null)
        {
            Debug.LogError("EggUI not set up correctly. Please assign the target and textComponent in the Inspector.");
        }
    }

    private void Update()
    {
        if (mainCamera == null || textComponent == null)
            return;

        // Make the UI face the camera (billboarding)
        transform.LookAt(transform.position + mainCamera.transform.rotation * Vector3.forward,
                         mainCamera.transform.rotation * Vector3.up);
    }

    private string AddNewLine(string text)
    {
        // Split the input text by spaces
        string[] words = text.Split(' ');

        // Insert a new line after the first word
        for (int i = 1; i < words.Length; i++)
        {
            // If the word contains characters other than whitespace
            if (!string.IsNullOrWhiteSpace(words[i]))
            {
                words[i] = "\n" + words[i];
                break;
            }
        }

        // Join the words back together
        string newText = string.Join(" ", words);

        return newText;
    }

    public void SetJerseyNumber(string jerseyNumber)
    {
        this.jerseyNumber = jerseyNumber;
    }

    public void SetPlayerName(string playerName)
    {
        // Add a new line to the text
        string newText = AddNewLine(playerName);

        // Set the text
        this.playerName = newText;
    }

    public void SetTextPlayerName()
    {
        // Set the text
        textComponent.text = playerName;
    }

    public void SetTextJerseyNumber()
    {
        // Set the text
        textComponent.text = jerseyNumber;
    }

    public void SetEmptyText()
    {
        // Set the text
        textComponent.text = "";
    }
}
