import { Canvas } from "@/components/Canvas";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-blue-50 to-purple-50 flex items-center">
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
        </div>

        <div className="w-1/2 relative">
          <div className="mx-auto relative flex flex-col items-center">
            <span
              className="absolute -top-8 text-sm font-medium text-indigo-600 bg-white/70 backdrop-blur-sm 
                           px-4 py-1.5 rounded-full shadow-sm inline-block animate-bounce z-10"
            >
              ✨ Draw Here ✨
            </span>
            <Canvas />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
