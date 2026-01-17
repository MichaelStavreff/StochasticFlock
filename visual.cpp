#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "simulation.hpp"

struct Simulation1d;
// struct Simulation2d;

sf::Font loadFont(std::string filename)
{
    sf::Font font;
    if (!font.loadFromFile(filename))
    {
        throw std::invalid_argument("Failed to load font: " + filename);
    }
    return font;
}

void init_simulation(Simulation1d &simulation)
{
    int iterations{};
    sf::Font font{loadFont("fonts/verdana.ttf")};
    sf::Text stats;
    sf::Text title;
    stats.setFont(font);
    stats.setCharacterSize(15);
    stats.setPosition(width * 0.90, 0);
    title.setFont(font);
    title.setCharacterSize(10);
    title.setString("StochasticFlock");
    title.setPosition(0, height * 0.97);

    sf::VertexArray plot_birds(sf::Quads, n_birds * 4);

    Eigen::Ref<Eigen::Matrix<double, n_birds, 3>> starting_positions{simulation.states};
    const Eigen::Vector<double, n_birds> y_plot{simulation.y_states};

    sf::RenderWindow window(sf::VideoMode(width, height), "StochasticFlock");
    window.setFramerateLimit(framerate);

    for (int i{0}; i < n_birds; ++i)
    {
        plot_birds[i].position = sf::Vector2f(width / 2 + starting_positions(i, 0), y_plot(i));
    }

    sf::VertexArray axis_line(sf::Lines, 2);
    sf::VertexArray lower_bound(sf::Lines, 2);
    sf::VertexArray upper_bound(sf::Lines, 2);

    axis_line[0].position = sf::Vector2f(0.f, height / 2);
    axis_line[1].position = sf::Vector2f(width, height / 2);
    lower_bound[0].position = sf::Vector2f(0.f, height * (0.5 + bound_offset));
    lower_bound[1].position = sf::Vector2f(width, height * (0.5 + bound_offset));
    upper_bound[0].position = sf::Vector2f(0.f, height * (0.5 - bound_offset));
    upper_bound[1].position = sf::Vector2f(width, height * (0.5 - bound_offset));
    lower_bound[0].color = sf::Color::Red;
    lower_bound[1].color = sf::Color::Red;
    upper_bound[0].color = sf::Color::Red;
    upper_bound[1].color = sf::Color::Red;

    while (window.isOpen() && iterations < rounds)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        // if (simulation.dimensions==1) {
        simulation.update_state(simulation.states);
        std::cout << simulation.full_states << '\n' << "After ^" << '\n';

        switch (simulation.steps_back)
        {
        case 0:
            std::cout << " ^" << '\n' << " States" << '\n';
            break;
        case 1:
            std::cout << "                                       ^" << '\n'
                      << "                                       States" << '\n';
            break;
        case 2:
            std::cout << "                                                                         ^" << '\n'
                      << "                                                                         States " << '\n';
            break;
        }
        std::cout << "-------------------------------------------------------" << '\n'
                  << "-------------------------------------------------------" << '\n';
        for (int i = 0; i < n_birds; ++i)
        {
            int v = i * 4;
            float x_pos = static_cast<float>(width / 2 + simulation.states(i, 0));
            float y_pos = static_cast<float>(simulation.y_states(i));
            float size = 1.0f;
            sf::Color birdColor = sf::Color::White;
            if (simulation.buffer(i, 2) > 0.5) // Leader
            {
                size = 4.0f;
                birdColor = sf::Color::Yellow;
            }

            plot_birds[v + 0].position = sf::Vector2f(x_pos, y_pos);
            plot_birds[v + 1].position = sf::Vector2f(x_pos + size, y_pos);
            plot_birds[v + 2].position = sf::Vector2f(x_pos + size, y_pos + size);
            plot_birds[v + 3].position = sf::Vector2f(x_pos, y_pos + size);

            for (int j = 0; j < 4; ++j)
            {
                plot_birds[v + j].color = birdColor;
            }
        }
        simulation.shift_back();
        std::cout << "SHIFT" << '\n';

        window.clear(sf::Color::Black);
        stats.setString(std::format("n={}\nIter: {}", n_birds, iterations));

        window.draw(stats);
        window.draw(title);
        window.draw(plot_birds);
        window.draw(axis_line);
        window.draw(upper_bound);
        window.draw(lower_bound);
        window.display();
        ++iterations;
    }
}

// void init_simulation(Simulation2d &simulation)

// {
//     int iterations{};
//     sf::Font font{loadFont("fonts/verdana.ttf")};
//     sf::Text stats;
//     sf::Text title;
//     stats.setFont(font);
//     stats.setCharacterSize(15);
//     stats.setPosition(width_2d * 0.90, 0);
//     title.setFont(font);
//     title.setCharacterSize(10);
//     title.setString("StochasticFlock");
//     title.setPosition(0, height_2d * 0.97);

//     sf::VertexArray plot_birds(sf::Points, n_birds);

//     Eigen::Matrix<float, n_birds, 2, Eigen::RowMajor>
//     starting_positions{simulation.states.leftCols(2).cast<float>()};

//     sf::RenderWindow window(sf::VideoMode(width_2d, height_2d), "StochasticFlock");
//     window.setFramerateLimit(framerate);

//     for (int i{0}; i < n_birds; ++i)
//     {
//         plot_birds[i].position = sf::Vector2f(width / 2 + starting_positions(i, 0), starting_positions(i, 1));
//     }

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

//     while (window.isOpen() && iterations < rounds)
//     {
//         sf::Event event;
//         while (window.pollEvent(event))
//         {
//             if (event.type == sf::Event::Closed)
//                 window.close();
//         }
//         // if (simulation.dimensions==1) {
//         simulation.update_state(simulation.states);

//         for (int i = 0; i < n_birds; ++i)
//         {
//             float x_pos = static_cast<float>(width_2d / 2 + starting_positions(i, 0));
//             float y_pos = static_cast<float>(height_2d / 2 + starting_positions(i, 1));

//             plot_birds[i].position = sf::Vector2f(x_pos, y_pos);

//             if (simulation.states(i, 2) == 1.0)
//             {
//                 plot_birds[i].color = sf::Color::Yellow; // Leaders
//             }
//             else
//             {
//                 plot_birds[i].color = sf::Color::White; // Followers
//             }
//         }

//         window.clear(sf::Color::Black);
//         stats.setString(std::format("n={}\nIter: {}", n_birds, iterations));

//         window.draw(stats);
//         window.draw(title);
//         window.draw(square);
//         window.draw(plot_birds);
//         window.draw(x_axis);
//         window.draw(y_axis);
//         window.display();
//         ++iterations;
//     }
// }
