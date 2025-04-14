import { useState } from "react";
import { Canvas, ModelType } from "@/components/Canvas";
import { Upload } from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Index = () => {
  const [selectedModel, setSelectedModel] = useState<ModelType>("mnist");
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.querySelector("canvas");
          const context = canvas?.getContext("2d");
          if (canvas && context) {
            const canvasWidth = canvas.clientWidth; // Rendered width
            const canvasHeight = canvas.clientHeight; // Rendered height

            // Clear the canvas
            context.clearRect(0, 0, canvas.width, canvas.height);

            // Calculate scaling to fit the image within the canvas
            const scaleX = canvasWidth / img.width;
            const scaleY = canvasHeight / img.height;
            const scale = Math.min(scaleX, scaleY); // Maintain aspect ratio

            // Calculate the position to center the image
            const offsetX = (canvasWidth - img.width * scale) / 2;
            const offsetY = (canvasHeight - img.height * scale) / 2;

            // Draw the image scaled and centered
            context.drawImage(
              img,
              0,
              0,
              img.width,
              img.height,
              offsetX,
              offsetY,
              img.width * scale,
              img.height * scale
            );
          }
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-100 via-blue-100 to-purple-100 flex items-center">
      <div className="container mx-auto px-10 flex flex-col md:flex-row items-center gap-10 py-8 md:py-0">
        <div className="w-full md:w-1/2 lg:px-16 text-center md:text-left">
          <h1
            className="font-bold bg-gradient-to-r from-purple-600 via-indigo-600 to-pink-500 text-transparent bg-clip-text 
                       tracking-tight leading-tight pb-4"
            style={{ fontSize: "clamp(3rem, 6vw, 6rem)" }} // Custom clamp for font size
          >
            Letter
            <br />
            Recognition
          </h1>
          <p
            className="text-gray-600 mt-2"
            style={{ fontSize: "clamp(1rem, 2vw, 2.5rem)" }} // Custom clamp for font size
          >
            Draw a letter and let AI predict what it is
          </p>
          <div className="mt-4 flex flex-col items-center md:items-start">
            <p className="text-sm font-medium text-gray-700 mb-2">
              Select Recognition Model:
            </p>
            <Tabs
              value={selectedModel}
              onValueChange={(value) => setSelectedModel(value as ModelType)}
              className="w-full max-w-md"
            >
              <TabsList className="grid grid-cols-4 w-full">
                <TabsTrigger value="mnist" className="text-xs sm:text-sm">
                  Mnist
                </TabsTrigger>
                <TabsTrigger value="emnist" className="text-xs sm:text-sm">
                  Emnist
                </TabsTrigger>
                <TabsTrigger
                  value="fashion-mnist"
                  className="text-xs sm:text-sm"
                >
                  Fashion Mnist
                </TabsTrigger>
                <TabsTrigger value="cifar-10" className="text-xs sm:text-sm">
                  Cifar-10
                </TabsTrigger>
              </TabsList>
            </Tabs>

            <div className="mt-3 mb-2 text-sm text-gray-600 max-w-md h-16">
              {selectedModel === "mnist" && (
                <p>
                  Our most accurate vision model capable of recognizing complex
                  or stylized handwritten digits.
                  <br />
                  Accuracy: 99.70%
                </p>
              )}
              {selectedModel === "emnist" && (
                <p>
                  A neural network with good accuracy for handwritten digits and
                  letters.
                  <br />
                  Accuracy: 86.66%
                </p>
              )}
              {selectedModel === "fashion-mnist" && (
                <p>
                  A neural network for recognizing clothing items.
                  <br />
                  Accuracy: 92.67%
                </p>
              )}
              {selectedModel === "cifar-10" && (
                <p>
                  A neural network for recognizing different objects.
                  <br />
                  Accuracy: 80.83%
                </p>
              )}
            </div>
          </div>
        </div>

        <div className="w-full md:w-1/2 flex justify-center md:justify-start mt-4 md:mt-0">
          <div className="w-[90%] relative flex flex-col items-center">
            <div className="absolute -top-8 flex items-center gap-4 z-10">
              <span
                className="text-sm font-medium text-indigo-600 bg-white/70 backdrop-blur-sm 
                           px-4 py-1.5 rounded-full shadow-sm inline-block animate-bounce"
              >
                ✨ Draw Here ✨
              </span>
              <label
                className="cursor-pointer bg-white/70 backdrop-blur-sm p-1.5 rounded-full shadow-sm 
                             hover:bg-white/90 transition-colors duration-200"
              >
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <Upload className="w-5 h-5 text-indigo-600" />
              </label>
            </div>
            <Canvas selectedModel={selectedModel} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
