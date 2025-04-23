import { Hero } from "@/components/Hero";
import { Projects } from "@/components/Projects";
import { Experience } from "@/components/Experience";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Hero />
      <Projects />
      <Experience />
    </div>
  );
};

export default Index;
