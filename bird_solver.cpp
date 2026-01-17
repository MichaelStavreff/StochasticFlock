
#include "constants.hpp"
#include "simulation.hpp"
struct Simulation1d;

#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void init_simulation(Simulation1d &simulation);

int main()
{
    Simulation1d Sim;

    // state matrix (cross sectional, all birds), delay matrix, matrix vector views
    // all followers at t=0
    // skip first delay amount of time to fill buffer

    // for (int i{}; i < 10; ++i)
    // {
    //     std::cout << "Round" << " " << i << '\n';
    //     std::cout << Sim.full_states << '\n' << "-------------------------------------------" << '\n';
    //     Sim.update_state(Sim.states);
    // }

    init_simulation(Sim);

    //     sf::RenderWindow window(sf::VideoMode(width_2d, height_2d), "StochasticFlock");
    //     window.setFramerateLimit(framerate);

    //     sf::VertexArray x_axis(sf::Lines, 2);
    //     sf::VertexArray y_axis(sf::Lines, 2);
    //     sf::RectangleShape square(sf::Vector2f(0.f, 0.f));

    //     x_axis[0].position = sf::Vector2f(0.f, height_2d * 0.5);
    //     x_axis[1].position = sf::Vector2f(width_2d, height_2d * 0.5);
    //     y_axis[0].position = sf::Vector2f(width_2d / 2, 0.f);
    //     y_axis[1].position = sf::Vector2f(width_2d / 2, height_2d);
    //     square.setSize(sf::Vector2f(box_size, box_size));
    //     square.setPosition(sf::Vector2f(width_2d * 0.5 - box_size * 0.5, height_2d * 0.5 - box_size * 0.5));
    //     square.setFillColor(sf::Color::Black);
    //     square.setOutlineThickness(1.f);
    //     square.setOutlineColor(sf::Color(92, 179, 219));

    //     while (window.isOpen())
    //     {
    //         sf::Event event;
    //         while (window.pollEvent(event))
    //         {
    //             if (event.type == sf::Event::Closed)
    //                 window.close();
    //         }

    //         window.clear(sf::Color::Black);

    //         window.draw(square);
    //         window.draw(y_axis);
    //         window.draw(x_axis);
    //         window.display();
    //     }
}
