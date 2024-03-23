using UnityEngine;

public class PitchControl : MonoBehaviour
{
    private Material squareMaterial; // Material to apply to squares

    [SerializeField] private GameObject pitchPrefab; // Plane to apply the texture to

    private float[,] colorBrighness; // Data to generate the texture

    private GameObject pitch;
    private int rows = 68; // Number of rows in the grid
    private int cols = 105; // Number of columns in the grid

    private Color color = Color.blue; // Color to apply to the texture

    public void AddPlaneAndTexture()
    {
        // Create a new plane and apply a texture to it
        squareMaterial = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        Vector3 position = new Vector3(52.5f, 0.01f, 34f);
        pitch = Instantiate(pitchPrefab, position, Quaternion.identity);
        Texture2D texture = GenerateTexture();
        ApplyTexture(texture);
    }

    public void UpdatePitchControlTexture()
    {
        // Generate a new texture
        Texture2D texture = GenerateTexture();

        // Apply the new texture
        ApplyTexture(texture);
    }
    public void UpdatePitchControlTexture(Vector2[] playerPositions)
    {

        for (int i = 0; i < playerPositions.Length; i++)
        {
            Debug.Log("Player " + i + " position: " + playerPositions[i]);
        }

        Debug.Log("Before json\n" + playerPositions);
        string json = JsonUtility.ToJson(playerPositions);
        Debug.Log("After json\n" + json);

        // Update the pitch data
        // colorBrighness = GetColors(playerPositions);


        // // Generate a new texture
        // Texture2D texture = GenerateTexture(colorBrighness);

        // // Apply the new texture
        // ApplyTexture(texture);
    }

    public void RemovePlaneAndTexture()
    {
        // Remove the plane and texture
        Destroy(pitch);
    }

    Texture2D GenerateTexture()
    {
        Texture2D texture = new Texture2D(cols, rows);
        // texture.filterMode = FilterMode.Point;


        // Generate colors for each pixel
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                Color color = GenerateRandomColor(); // Or any method to generate colors as needed
                texture.SetPixel(x, y, color);
            }
        }

        texture.Apply(); // Apply changes to the texture

        return texture;
    }

    Texture2D GenerateTexture(float[,] data)
    {
        Texture2D texture = new Texture2D(cols, rows);
        texture.filterMode = FilterMode.Point;

        // Generate colors for each pixel based on the data
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                Color color = GenerateColorFromData(data[x, y]); // Or any method to generate colors based on data
                texture.SetPixel(x, y, color);
            }
        }

        texture.Apply(); // Apply changes to the texture

        return texture;
    }

    private void ApplyTexture(Texture2D texture)
    {
        // Apply the texture to the material
        squareMaterial.mainTexture = texture;

        // Apply the material to each square GameObject
        GameObject[] squares = GameObject.FindGameObjectsWithTag("PitchOverlay"); // Assuming squares have a "Square" tag
        foreach (GameObject square in squares)
        {
            Renderer renderer = square.GetComponent<Renderer>();
            renderer.sharedMaterial = squareMaterial;
        }
    }

    private Color GenerateRandomColor()
    {
        // Generate a random color
        return new Color(Random.value, Random.value, Random.value);
    }

    private Color GenerateColorFromData(float value)
    {
        // Generate a color based on the data value
        return Utils.ChangeBrightness(color, value);
    }

    private float[,] GetColors(Vector2[] playerPositions)
    {
        // Get colors for each data point from Thiago's script
        float[,] colors = new float[cols, rows];
        return colors;
    }

    private string Vector2ToJson(Vector2[] vector)
    {
        // Convert Vector2 array to JSON
        return JsonUtility.ToJson(vector);
    }

}