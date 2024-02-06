using UnityEngine;

public class TerrainPitchColor : MonoBehaviour
{
    int textureWidth = 256;
    int textureHeight = 256;

    public int pitchLength = 68;

    public int pitchWidth = 105;
    public Color stripeColor1 = new Color(48f / 255f, 96f / 255f, 48f / 255f);
    public Color stripeColor2 = new Color(96f / 255f, 128f / 255f, 56f / 255f);

    public Texture2D texture_lines;

    public int stripes = 16; // number of stripes

    void Start()
    {
        // Get the terrain component
        Terrain terrain = GetComponent<Terrain>();
        Debug.Log(terrain.name);

        // Apply the stripe texture to the terrain
        ApplyStripeTextureToTerrain(terrain);
    }

    void ApplyStripeTextureToTerrain(Terrain terrain)
    {
        // Get the terrain data
        TerrainData terrainData = terrain.terrainData;

        // Create a new texture
        Texture2D texture_grass = GenerateGrassTexture();

        // Apply the texture to the terrain's alphamap
        ApplyTerrainMaterial(terrainData, texture_grass, 1);
        ApplyTerrainMaterial(terrainData, texture_lines, 0);
    }


    Texture2D GenerateLinesTexture()
    {
        Texture2D linesTexture = Resources.Load<Texture2D>(""); // Make sure to place the texture in the Resources folder

        if (linesTexture == null)
        {
            Debug.LogError("Lines texture not found. Make sure to create and import the texture.");
        }

        return linesTexture;
    }
    // Creates the striped pitch texture
    Texture2D GenerateGrassTexture()
    {
        // Create a new texture
        Texture2D texture = new Texture2D(textureWidth, textureHeight);

        // Set the filter mode to point to get hard, pixelated edges
        texture.filterMode = FilterMode.Point;

        for (int x = 0; x < textureWidth; x++)
        {
            for (int y = 0; y < textureHeight; y++)
            {
                // Determine the stripe color based on the x position
                Color stripeColor = (x / (textureWidth / stripes)) % 2 == 0 ? stripeColor1 : stripeColor2;

                texture.SetPixel(x, y, stripeColor);
            }
        }

        // Apply changes
        texture.Apply();

        return texture;
    }

    void ApplyTerrainMaterial(TerrainData terrainData, Texture2D texture, int layer)
    {
        TerrainLayer[] terrainLayers = terrainData.terrainLayers;

        if (terrainLayers != null && terrainLayers.Length > 0)
        {
            Vector2 tileSize = new Vector2(pitchWidth, pitchLength);
            // Set the material of the first terrain layer
            terrainLayers[layer].tileSize = tileSize;
            terrainLayers[layer].diffuseTexture = texture; // Assuming the material has a main texture

            // Update the terrain data with the modified terrain layers
            terrainData.terrainLayers = terrainLayers;
        }
        else
        {
            Debug.LogError("Terrain layers not found or empty. Make sure your terrain has at least one layer.");
        }
        // Set the material to the terrain renderer
    }


}
