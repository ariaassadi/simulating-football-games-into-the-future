using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal; // Import URP namespace

public class PitchColor : MonoBehaviour
{
    public int textureWidth = 256;
    public int textureHeight = 256;
    public Color stripeColor1 = new Color(48f / 255f, 96f / 255f, 48f / 255f);
    public Color stripeColor2 = new Color(96f / 255f, 128f / 255f, 56f / 255f);
    public int stripeWidth = 16; // Adjust this value for more stripes

    void Start()
    {
        // Create a new texture
        Texture2D texture = GenerateTexture();

        // Create a material and assign the texture
        Material material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        material.mainTexture = texture;

        // Apply the material to the GameObject's renderer
        Renderer renderer = GetComponent<Renderer>();
        renderer.material = material;
        //material;
    }

    Texture2D GenerateTexture()
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
                Color stripeColor = (x / stripeWidth) % 2 == 0 ? stripeColor1 : stripeColor2;

                texture.SetPixel(x, y, stripeColor);
            }
        }

        // Apply changes
        texture.Apply();

        return texture;
    }

    // void Start()
    // {
    //     // Create a new texture
    //     Texture2D texture = new Texture2D(textureWidth, textureHeight);

    //     // Set the filter mode to point to get hard, pixelated edges
    //     texture.filterMode = FilterMode.Point;

    //     for (int x = 0; x < textureWidth; x++)
    //     {
    //         for (int y = 0; y < textureHeight; y++)
    //         {
    //             // Determine the stripe color based on the x position
    //             Color stripeColor = (x / stripeWidth) % 2 == 0 ? stripeColor1 : stripeColor2;

    //             texture.SetPixel(x, y, stripeColor);
    //         }
    //     }

    //     // Apply changes and assign the texture to a material
    //     texture.Apply();
    //     Renderer renderer = GetComponent<Renderer>();
    //     renderer.material.mainTexture = texture;
    // }
}

