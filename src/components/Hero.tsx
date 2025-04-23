import { Github, Mail, Linkedin } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Hero = () => {
  return (
    <section className="min-h-[80vh] flex flex-col items-center justify-center text-center px-4">
      <img
        src="https://github.com/ghost.png"
        alt="Profile"
        className="w-32 h-32 rounded-full mb-8 animate-fadeIn"
      />
      <h1 className="text-4xl md:text-5xl font-bold mb-4 animate-fadeIn">
        John Doe
      </h1>
      <p className="text-xl text-muted-foreground mb-8 max-w-2xl animate-fadeIn">
        Full Stack Developer passionate about building beautiful interfaces & experiences
      </p>
      <div className="flex gap-4 animate-fadeIn">
        <Button variant="outline" size="lg" className="hover-scale">
          <Github className="mr-2 h-5 w-5" />
          GitHub
        </Button>
        <Button variant="outline" size="lg" className="hover-scale">
          <Linkedin className="mr-2 h-5 w-5" />
          LinkedIn
        </Button>
        <Button variant="outline" size="lg" className="hover-scale">
          <Mail className="mr-2 h-5 w-5" />
          Contact
        </Button>
      </div>
    </section>
  );
};
