Shader "Unlit/Highlight"
{
    Properties
    {
        _Radius ("Radius", Range(0.0, 1.0)) = 0.5
        _LineWidth ("Line Width", Range(0.0, 0.1)) = 0.01
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : POSITION;
            };

            float _Radius;
            float _LineWidth;

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Calculate distance from the center of the screen
                float dist = length(0.5 - i.pos.xy);

                // Create a circle by checking if the distance is within a certain range
                float circle = smoothstep(_Radius - _LineWidth, _Radius, dist);
                
                // Subtract a smaller circle inside the main circle to create an outline
                float outline = smoothstep(_Radius, _Radius + _LineWidth, dist);

                // Combine circle and outline
                float result = circle - outline;

                return fixed4(1, 1, 1, result);
            }
            ENDCG
        }
    }
}