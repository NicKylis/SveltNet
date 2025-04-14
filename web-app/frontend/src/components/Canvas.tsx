import { useState, useRef, useEffect } from "react";
import { RefreshCw } from "lucide-react";

export type ModelType = "mnist" | "emnist" | "fashion-mnist" | "cifar-10";

interface CanvasProps {
  selectedModel: ModelType;
}

export const Canvas = ({ selectedModel }: CanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const canvasImageRef = useRef<string | null>(null);
  const [containerSize, setContainerSize] = useState<{
    width: number;
    height: number;
  } | null>(null);

  useEffect(() => {
    if (canvasContainerRef.current && !containerSize) {
      const width = canvasContainerRef.current.offsetWidth;
      const height = canvasContainerRef.current.offsetHeight;
      setContainerSize({ width, height });
    }
  }, [containerSize]);

  const setupCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const container = canvasContainerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.scale(dpr, dpr);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = 12;
    context.strokeStyle = "#000000";
    contextRef.current = context;

    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);

    if (canvasImageRef.current) {
      const img = new Image();
      img.onload = () => {
        context.drawImage(img, 0, 0, rect.width, rect.height);
      };
      img.src = canvasImageRef.current;
    }
  };

  useEffect(() => {
    setupCanvas();

    const handleResize = () => {
      if (canvasRef.current && contextRef.current) {
        canvasImageRef.current = canvasRef.current.toDataURL();
      }

      setupCanvas();
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  const getCoordinates = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();

    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;

    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  };

  const startDrawing = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    setIsDrawing(true);
    if (!contextRef.current) return;

    const { x, y } = getCoordinates(e);
    contextRef.current.beginPath();
    contextRef.current.moveTo(x, y);
  };

  const draw = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    if (!isDrawing || !contextRef.current) return;

    const { x, y } = getCoordinates(e);
    contextRef.current.lineTo(x, y);
    contextRef.current.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    if (contextRef.current) {
      contextRef.current.closePath();
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    if (!canvas || !context) return;

    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);

    canvasImageRef.current = null;
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    setPrediction(null);
    console.log(`Predicting with model: ${selectedModel}`);
    // Capture the image from the canvas
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dataURL = canvas.toDataURL("image/png"); // Get the base64 string representation of the image
    const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:5000";
    try {
      // Send the image to the Flask API
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json", // Set the content type to JSON
        },
        body: JSON.stringify({
          image: dataURL, // Send the base64 string in JSON
          model: selectedModel, // Include the selected model type
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      console.log("Prediction:", data.prediction); // Handle the predicted result as needed
      setPrediction(data.prediction);
      setIsPredicting(false);
    } catch (error) {
      console.error("Error during prediction:", error);
      setIsPredicting(false);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-6">
      <div
        ref={canvasContainerRef}
        className="relative w-[80%] aspect-square bg-white rounded-3xl shadow-[0_0_40px_rgba(99,102,241,0.15)] 
                  overflow-hidden border-4 border-indigo-100/50 transition-all duration-300 hover:shadow-[0_0_60px_rgba(99,102,241,0.25)]"
        style={
          containerSize
            ? {
                width: `${containerSize.width}px`,
                height: `${containerSize.height}px`,
              }
            : undefined
        }
      >
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="touch-none cursor-crosshair"
        />
      </div>

      <div className="flex gap-4 w-full justify-center">
        <button
          onClick={clearCanvas}
          className="group flex items-center gap-2 px-6 py-3 bg-white text-indigo-600 rounded-xl shadow-sm 
                   hover:bg-indigo-50 active:bg-indigo-100 transition-all duration-200 ease-in-out
                   border-2 border-indigo-100"
        >
          <RefreshCw className="w-5 h-5 transition-transform group-hover:rotate-180 duration-500" />
          <span>Clear</span>
        </button>

        <button
          onClick={handlePredict}
          disabled={isPredicting}
          className={`px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl shadow-sm
                    hover:from-indigo-500 hover:to-purple-500 active:from-indigo-700 active:to-purple-700 
                    transition-all duration-200 ease-in-out font-medium
                    ${isPredicting ? "opacity-75 cursor-not-allowed" : ""}`}
        >
          {isPredicting ? "✨ Predicting..." : "✨ Predict"}
        </button>
      </div>
      {/* Display prediction result */}
      {prediction !== null && (
        <div className="mt-4 text-xl font-semibold text-indigo-600">
          Predicted: {prediction}
        </div>
      )}
    </div>
  );
};
