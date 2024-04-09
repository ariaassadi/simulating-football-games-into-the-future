using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal; // Import URP namespace
using System;

public class PitchColor : MonoBehaviour
{
    private int textureWidth = 256;
    private int textureHeight = 256;
    public Color stripeColor1 = new Color(48f / 255f, 96f / 255f, 48f / 255f);
    public Color stripeColor2 = new Color(96f / 255f, 128f / 255f, 56f / 255f);

    public Color lineColor = Color.white; // Color of the lines
    public int stripeWidth = 16; // Adjust this value for more stripes

    private int lineThickness = 200;

    private MeshRenderer meshRenderer;

    public Texture2D linesTexturePNG;

    void Start()
    {
        // Create a new texture
        Texture2D grassTexture = GenerateGrassTexture();
        // Texture2D linesTexture = GenerateLinesTexture();

        // Create a material and assign the texture
        Material grassMaterial = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        grassMaterial.mainTexture = grassTexture;

        Material linesMaterial = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        Debug.Log(linesMaterial.shader);
        // Set surface type to Transparent
        linesMaterial.mainTexture = linesTexturePNG;
        // linesMaterial.SetFloat("_Surface", 1.0f);

        // Apply the material to the GameObject's renderer
        meshRenderer = GetComponent<MeshRenderer>();

        if (meshRenderer)
        {
            Material[] materials = meshRenderer.materials;

            if (materials != null && materials.Length > 0)
            {
                // Example: Change the first material to a new material
                // Replace "newMaterial" with the material you want to assign
                materials[0] = grassMaterial;
                // materials[1] = linesMaterial;

                // Assign the modified materials array back to the MeshRenderer
                meshRenderer.materials = materials;
            }
        }
    }

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
                Color stripeColor = (x / stripeWidth) % 2 == 0 ? stripeColor1 : stripeColor2;

                texture.SetPixel(x, y, stripeColor);
            }
        }

        // Apply changes
        texture.Apply();

        return texture;
    }

    Texture2D GenerateLinesTexture()
    {
        Texture2D texture = new Texture2D(textureWidth, textureHeight);
        texture.filterMode = FilterMode.Point;

        Color[] colors = new Color[textureWidth * textureHeight];
        for (int i = 0; i < colors.Length; i++)
        {
            colors[i] = Color.clear;
        }
        texture.SetPixels(colors);

        // Draw soccer pitch markings
        DrawSoccerPitchLines(texture);

        texture.Apply();
        return texture;
    }

    void DrawSoccerPitchLines(Texture2D texture)
    {
        // Draw soccer pitch markings
        DrawCenterCircle(texture);
        // DrawPenaltyAreas(texture);
        // DrawGoalBoxes(texture);
        // DrawTouchlines(texture);
        // DrawGoalLines(texture);
    }

    void DrawCenterCircle(Texture2D texture)
    {
        // Draw center circle
        int centerX = textureWidth / 2;
        int centerY = textureHeight / 2;
        int radius = textureHeight / 8;

        DrawCircle(texture, centerX, centerY, radius, lineColor);
    }

    void DrawPenaltyAreas(Texture2D texture)
    {
        // Draw penalty areas
        int penaltyAreaWidth = textureWidth / 4;
        int penaltyAreaHeight = textureHeight / 3;

        int leftPenaltyX = (textureWidth - penaltyAreaWidth) / 2;
        int rightPenaltyX = leftPenaltyX + penaltyAreaWidth;
        int penaltyY = (textureHeight - penaltyAreaHeight) / 2;

        DrawRect(texture, leftPenaltyX, penaltyY, penaltyAreaWidth, penaltyAreaHeight, lineColor);
        DrawRect(texture, rightPenaltyX, penaltyY, penaltyAreaWidth, penaltyAreaHeight, lineColor);
    }

    void DrawGoalBoxes(Texture2D texture)
    {
        // Draw goal boxes
        int goalBoxWidth = textureWidth / 8;
        int goalBoxHeight = textureHeight / 6;

        int leftGoalX = textureWidth / 8;
        int rightGoalX = textureWidth - (textureWidth / 8);
        int goalBoxY = (textureHeight - goalBoxHeight) / 2;

        DrawRect(texture, leftGoalX, goalBoxY, goalBoxWidth, goalBoxHeight, lineColor);
        DrawRect(texture, rightGoalX, goalBoxY, goalBoxWidth, goalBoxHeight, lineColor);
    }

    void DrawTouchlines(Texture2D texture)
    {
        // Draw touchlines (sidelines)
        int lineWidth = 2; // Width of touchlines

        int topY = textureHeight - lineWidth;
        int bottomY = lineWidth;

        DrawRect(texture, 0, topY, textureWidth, lineWidth, lineColor);
        DrawRect(texture, 0, bottomY, textureWidth, lineWidth, lineColor);
    }

    void DrawGoalLines(Texture2D texture)
    {
        // Draw goal lines (end lines)
        int lineWidth = 2; // Width of goal lines

        int leftX = lineWidth;
        int rightX = textureWidth - lineWidth;

        DrawRect(texture, leftX, 0, lineWidth, textureHeight, lineColor);
        DrawRect(texture, rightX, 0, lineWidth, textureHeight, lineColor);
    }

    void DrawCircle(Texture2D texture, int centerX, int centerY, int radius, Color color)
    {
        texture.filterMode = FilterMode.Trilinear;

        for (int x = centerX - radius; x <= centerX + radius; x++)
        {
            for (int y = centerY - radius; y <= centerY + radius; y++)
            {
                if (Math.Abs(((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY) - radius * radius)) <= lineThickness)
                {
                    texture.SetPixel(x, y, color);
                }
            }
        }
    }

    void DrawRect(Texture2D texture, int x, int y, int width, int height, Color color)
    {
        for (int i = x; i < x + width; i++)
        {
            for (int j = y; j < y + height; j++)
            {
                texture.SetPixel(i, j, color);
            }
        }
    }

}

